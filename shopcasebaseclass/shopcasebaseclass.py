

import os
import json
import zipfile
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import ByteString, Union, Dict, List
from collections import namedtuple
from io import BytesIO

import pandas as pd
import numpy as np
import yaml
from graphviz import Digraph

try:
    from pyshop import ShopSession  # pylint: disable=import-error
except:
    ShopSessionError = NameError('Cannot access ShopSession : pyshop is not available')
    class ShopSession:
        def __init__(self):
            raise ShopSessionError
        def __getattr__(self, attr):
            raise ShopSessionError
        def __setattr__(self, attr, value):
            raise ShopSessionError

    #print('When loading ShopCaseBaseClass.py: cannot import pyshop')


class ShopCaseBaseClass:
    """ Parent class for local ShopCase implementation.

    ## What this is
    ShopCase is a data structure for SHOP model data. The term 'case' in ShopCase
    refers to a consistent data set that can form a single ShopSession instance. 

    When saved to disk, the convention is to use the file extention .shop.zip,
    e.g. 'mycase.shop.zip'. The zip format is used to compress the data, and 
    the .zip part of the file extention is used to facilitate manual inspection 
    of the files without extracting the archive. the .shop. part of the file 
    extention is meant to separate these standardized files from any other 
    zip files.

    ## Background
    ShopCase is developed to overcome limitations in the traditional files used 
    by SHOP (ASCII files etc.).

    The foreseen benefits of ShopCase compared to SHOP cases based on ASCII 
    files are:
    - machine readable based on common standards and data structures
    - human readable, where all object attributes are explicitly named
    - can be inspected and modified independent of shop/pyshop availability and
      ShopSession initiation
    - case data can be serialized and passed between different services, thus 
      enbling a functional split between case creation, optimization, 
      evaluation and visualization
    - the complete inputs and outputs of a Shop optimization is stored in a single 
      consistent data structure or file, enabling reproducability
    - the data is compressed when saved to a file

    ## Inheriting from this class
    This class defines the internal data structure of a ShopCase instance, and 
    some methods that should NOT be overridden (to ensure tha case data is valid
    between different implementations of ShopCase):
    - to_shopsession : method for generating a ShopSession instance based on a ShopCase
    - from_shopsession : method for initiating a ShopCase instance from a ShopSession
    - to_file : exporting ShopCase data to a .shop.zip file
    - from_file : initiation a ShopCase instance from a .shop.zip file

    In addition the class defines methods that can be overridden based on 
    local implementation:
    - run : method for running the ShopCase
    - copy : make a deep copy of the case

    Classes derived from this base class can be made to contain custom fields 
    under <ShopCase>.case that will be preserved through copying and running the 
    ShopCase instance, but these data must be json serializable. Each such 
    field will be stored as a separate .yaml file when the case is saved to a 
    .shop.zip file.

    Proposal for class design guidelines:
    - Most methods should return self, to facilitate method chaining, e.g. 
      ShopCase('mycase.shop.zip').run().
    - The run method should insert the optimization results into the current 
      instance (and all previous results removed). If you want to preserve the 
      original state, then use <ShopCase>.copy().run().

    """
    
    def __init__(self, source):
        self.case = None
        self.log_func = print

        if isinstance(source, ShopSession):
            self._from_shopsession(source)
        elif isinstance(source, dict):
            self.case = source
        elif isinstance(source, bytes):
            self._from_bytestring(source)
        elif isinstance(source, str) and source[0] == '{':
            self._from_json(source)
        elif isinstance(source, (str, Path)) and (Path(source) / 'model.yaml').exists():
            self._from_dir(source)
        elif isinstance(source, (str, Path)) and '.shop.zip' in Path(source).name:
            self._from_file(source)


        
    def run(
        self, 
        shopdir=None, 
        **shopsession_kwargs) -> ShopSession:
        """ Run the ShopCase in pyshop, and update self with the new results. """

        # Move to a temporary directory
        _cwd = os.getcwd()
        if shopdir is None:
            shopdir = tempfile.TemporaryDirectory(prefix='shopcase_').name
            os.mkdir(shopdir)
            os.chdir(shopdir)
        else: 
            shopdir = _cwd()
        self.shopdir = shopdir

        # Save the cut files (if any)
        if 'cutfiles' in self.case:
            for filename, filecontent in self.case['cutfiles'].items():
                Path(filename).write_text(filecontent)

        # Run SHOP
        shop = self.to_shopsession(**shopsession_kwargs)
        shop.model.update()
        for c in self.case['commands']:
            self.log_func(c)
            shop.shop_api.ExecuteCommand(c)
        shop.model.update()

        # Preserve custom fields in self.case
        _old_case = self.case 
        self._from_shopsession(shop)
        for i in _old_case:
            if i not in self.case:
                self.case[i] = _old_case[i]

        # Save the logs
        self.case['logs'] = {}
        self.case['logs']['shop'] = shop.get_messages()
        if Path('cplex.log').exists():
            self.case['logs']['cplex'] = Path('cplex.log').read_text()

        # Return to previous working directory
        os.chdir(_cwd)

        return shop

    def open_shopdir(self):
        if hasattr(self, 'shopdir'):
            os.startfile(self.shopdir)      

    def copy(self):      
        """ Create a new ShopCase instance from a deep copy of self.case."""
        return self.__class__(deepcopy(self.case))

    def to_shopsession(self, **shopsession_kwargs) -> ShopSession:
        
        shop = ShopSession(**shopsession_kwargs)

        commands_to_call_before_object_creation = ['set newgate /off']
        for c in commands_to_call_before_object_creation:
            if c in self.case['commands']:
                self.log_func(f'Calling command "{c}"')
                shop.shop_api.ExecuteCommand(c)

        model = self.case['model']

        # Set time resolution
        shop.set_time_resolution(**self.case['time'])
        
        ## Create the objects
        scenarios = []
        for obj_type in model:
            if shop.shop_api.GetObjectInfo(obj_type, 'isInput') == 'False':
                continue
            for obj_name in model[obj_type]:
                # Keep track of scenario id in order to add them in the correct order later on
                if obj_type == 'scenario':
                    scenarios.append((model['scenario'][obj_name]['scenario_id'], obj_name))
                    continue
                shop.shop_api.AddObject(obj_type, obj_name)

        ## Create scenarios, make sure the scenarios are added in the correct order
        scenarios = sorted(scenarios)
        for _, obj_name in scenarios:
            shop.shop_api.AddObject('scenario', obj_name)

        ## Add object attributes
        for obj_type, obj_name, attr, value in self._crawl_model():
            if shop.shop_api.GetObjectInfo(obj_type, 'isInput') == 'False':
                continue
            if shop.shop_api.GetAttributeInfo(obj_type, attr, 'isInput') == 'True':
                shop.model[obj_type][obj_name][attr].set(value)
            
        # Connect objects
        conn = self.case['connections']
        for to_type in conn:
            for to_name in conn[to_type]:
                for connection_map in conn[to_type][to_name]:
                    try:
                        shop.shop_api.AddRelation(connection_map['upstream_obj_type'],
                                                  connection_map['upstream_obj_name'],
                                                  connection_map['connection_type'], to_type, to_name)

                    except ValueError:
                        self.log_func(f'to_shopsession : could not connect '
                                      f'from_type={connection_map["upstream_obj_type"]}, '
                                      f'from_name={connection_map["upstream_obj_name"]}, '
                                      f'conn_type={connection_map["connection_type"]}, '
                                      f'to_type={to_type}, '
                                      f'to_name={to_name}')
        #conn = self.case['connections']
        #for from_type in conn:
        #    for from_name in conn[from_type]:
        #        for to_type, to_name, conn_type in conn[from_type][from_name]:
        #            try:
        #                shop.shop_api.AddRelation(from_type, from_name, conn_type, to_type, to_name)
        #            except ValueError:
        #                self.log_func(f'to_shopsession : could not connect from_type={from_type}, from_name={from_name}, conn_type={conn_type}, to_type={to_type}, to_name={to_name}')

        return shop

    def diff(self, other: 'ShopCaseBaseClass', tolerance: float = 0.1) -> Dict:
        """ Compare the data in two ShopCases and return a dict with the same 
        structure as ShopCase.case indicating where the data is different.

        Returns an empty dict if the cases are identical (within the tolerance).
        
        The tolerance applies to data in pandas.Series, and is by default set
        to 0.1, and should be less than 1 to catch differences in binary values
        like unit commitment.
        """
    
        return self._compare(self.case, other.case, tolerance)

    @classmethod
    def _compare(cls, left, right, tolerance: float = 0.0):
        """ Compare two dictionaries. """
        
        diffs = {}
        
        for i in set(left) | set(right):
            if i not in right or i not in left:
                diffs[i] = True
            elif isinstance(left[i], dict):
                res = cls._compare(left[i], right[i], tolerance)
                if res:
                    diffs[i] = res
            else:
                status = cls._is_different(left[i], right[i], tolerance)
                if status:
                    diffs[i] = status
                    
        return diffs

    @classmethod           
    def _is_different(cls, i, j, tolerance: float = 0.0):
        """ Compare two objects. """

        if not type(i) is type(j):
            return True
        if isinstance(i, (int, float, str, pd.Timestamp)):
            return i != j
        if isinstance(i, pd.Series):
            try:
                # Dropping consecutive identical values before we compare data
                _i = i[i != i.shift(1)].dropna()
                _j = j[j != j.shift(1)].dropna()
                return ((_i - _j).abs().max() > tolerance) or (_i - _j).isna().any()
            except ValueError:
                raise ValueError
        if isinstance(i, dict):
            return cls._compare(i, j, tolerance)
        if isinstance(i, list):
            if len(i) != len(j):
                return True
            return any([cls._is_different(m, n, tolerance) for m, n in zip(i, j)])
        
        raise ValueError(f'Type not specified in is_equal : {i}={type(i)}')

    def drop_mc_data(self):
        """ Drop the large data sets to make it feasible to serialize data to disk. """
        
        def drop_attr_from_subdict(d, attr):
            for sub in d.values():
                if isinstance(sub, dict):
                    if attr in sub:
                        _ = sub.pop(attr)

        drop_attr_from_subdict(self.case['model']['plant'], 'best_profit_ac')
        drop_attr_from_subdict(self.case['model']['plant'], 'best_profit_mc')
        drop_attr_from_subdict(self.case['model']['plant'], 'best_profit_q')
        drop_attr_from_subdict(self.case['model']['generator'], 'best_profit_p')
        drop_attr_from_subdict(self.case['model']['generator'], 'best_profit_q')
        drop_attr_from_subdict(self.case['model']['generator'], 'average_cost')

        if 'unit_combination' in self.case['model']:
            _ = self.case['model'].pop('unit_combination')

    def get_txys(self):
        data = {}
        for obj_type, obj_name, attr, value in self._crawl_model():
            if isinstance(value, pd.Series) and isinstance(value.index, pd.DatetimeIndex):
                name = '|'.join([obj_type, obj_name, attr]).lower()
                data[name] = value
        df = pd.DataFrame(data)

        # Flip the sign to more intuitive value
        patterns = [
            'incr_cost_nok_mw',
            '|sale|mw'
        ]
        for c in df.columns:
            if any([p in c for p in patterns]):
                df[c] *= -1

        return df

    def show_topology(self) -> Digraph:
            
        dot = Digraph(comment='SHOP topology')
        
        Node = namedtuple('Node', ['type', 'label', 'name'])
        Edge = namedtuple('Edge', ['relation', 'start', 'end'])
        
        default_node_attrs = {'shape': 'ellipse', 'fillcolor': '#ffffff', 'style': ''}
        node_attrs = {
            'plant': {'shape': 'box', 'fillcolor': '#FF9999', 'style': 'filled'},
            'reservoir': {'shape': 'invtriangle', 'fillcolor': '#99FFFF', 'style': 'filled'},
            'junction': {'shape': 'point', 'fillcolor': '#ffffff', 'style': ''},
            'junction_gate': {'shape': 'point', 'fillcolor': '#ffffff', 'style': ''}
        }
        
        def get_edge_style(e: Edge):
            if 'gate' in [e.start.type, e.end.type] and e.relation == 'connection_standard':
                return 'dashed'
            return 'solid'  
        
        nodes = set()
        edges = set()
        
        for obj_type in self.case['connections']:
            for obj_name, conn in self.case['connections'][obj_type].items():
                node1 = Node(type=obj_type, label=obj_name, name=f'{obj_type}_{obj_name}')
                nodes.add(node1)
                
                for obj_type2, obj_name2, conn_type in conn:
                    node2 = Node(type=obj_type2, label=obj_name2, name=f'{obj_type2}_{obj_name2}')
                    edge = Edge(relation=conn_type, start=node1, end=node2)         
                    nodes.add(node2)
                    edges.add(edge)
                
        for n in nodes:
            dot.node(n.name, label=n.label, **node_attrs.get(n.type, default_node_attrs))
                
        for e in edges:
            dot.edge(e.start.name, e.end.name, style=get_edge_style(e))
            
        return dot

    def to_file(self, filename: Union[str, Path] = 'case.shop.zip'):
        self.drop_mc_data()  # Drop bestprofit and other marginal cost data as it inflates the model file too much
        with open(filename, 'wb') as f:
            f.write(self._to_bytestring())
        return Path(filename).absolute()

    def _to_bytestring(self) -> ByteString:
        b = BytesIO()
        with zipfile.ZipFile(b, 'w', zipfile.ZIP_DEFLATED) as f:
            for key, value in self._get_dict_with_json_types().items():
                f.writestr(f'{key}.yaml', yaml.dump(value, allow_unicode=True, encoding='UTF-8', sort_keys=False)) #.encode('utf-8')
        return b.getvalue()

    def to_json(self):
        return json.dumps(self._get_dict_with_json_types(), ensure_ascii=False)

    def to_yaml_files(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        for key, value in self._get_dict_with_json_types().items():
            with open(path / f'{key}.yaml', 'bw') as f:
                f.write(yaml.safe_dump(value, allow_unicode=True, encoding='UTF-8', sort_keys=False)) #.encode('utf-8')
        return path

    def _from_json(self, s: str):
        self.case = json.loads(s)
        self._convert_to_pyshop_types()

    def _from_file(self, filename: Union[str, Path]):
        with open(filename, 'rb') as f:
            b = f.read()
        self._from_bytestring(b)
    
    def _from_bytestring(self, b: ByteString) -> Dict:
        z = zipfile.ZipFile(BytesIO(b), 'r')
        yaml_files = [i.filename for i in z.filelist]
        self.case = dict([(i.replace('.yaml', ''), yaml.safe_load(z.read(i))) for i in yaml_files])
        self._convert_to_pyshop_types()
    
    def _from_dir(self, path: str) -> Dict:
        """ Initialize ShopCase from a set of yaml files in 'path' (i.e. an 
        unzipped *.shop.zip file).
        """
        self.case = {}
        for i in Path(path).iterdir():
            with open(i, 'r', encoding='UTF-8') as f:
                self.case[i.name.replace('.yaml', '')] = yaml.safe_load(f)
        self._convert_to_pyshop_types()

    def _crawl_model(self):
        for obj_type in self.case['model']:
            for obj_name in self.case['model'][obj_type]:
                for attr in self.case['model'][obj_type][obj_name]:
                    value = self.case['model'][obj_type][obj_name][attr]
                    yield obj_type, obj_name, attr, value
        
    def _from_shopsession(self, shop: ShopSession):
        d = {
            'model': self._get_model(shop), 
            'connections': self._get_connections(shop),
            'time': shop.get_time_resolution(),
            'commands': shop.get_executed_commands()
            }

        self.case = deepcopy(d)  # Necessary?

    def _get_model(self, shop: ShopSession) -> Dict:
        """ Dump all data in ShopSession.model to nested dict. """

        shop.model.update()
        
        obj_list = sorted(
            zip(shop.shop_api.GetObjectTypesInSystem(), 
                shop.shop_api.GetObjectNamesInSystem()), 
            key=lambda x: x[0]+x[1]
        )
        
        model = dict()
        
        for obj_type, obj_name in obj_list:   
            try: 
                obj = shop.model[obj_type][obj_name]
            except AttributeError:
                self.log_func(f'_get_model : Could not get shop.model[{obj_type}][{obj_name}]')
                continue

            attrs = obj.datatype_dict
    
            for a in attrs:
                value = obj[a].get()
                if value is not None:
                    if obj_type not in model:
                        model[obj_type] = {}
                    if obj_name not in model[obj_type]:
                        model[obj_type][obj_name] = {}

                    model[obj_type][obj_name][a] = value
    
        return model   
            
    def _get_connections(self, shop: ShopSession) -> Dict:
        obj_list = list(
            zip(shop.shop_api.GetObjectTypesInSystem(), 
                shop.shop_api.GetObjectNamesInSystem())
        )
        obj_dicts = [{'obj_type': x[0], 'obj_name': x[1]} for x in obj_list]
        
        connections = {}
    
        for to_obj_type, to_obj_name in obj_list:
            if shop.shop_api.GetObjectInfo(to_obj_type, 'isInput') == 'False':
                continue

            input_relations = []
            for conn_type in shop.shop_api.GetValidRelationTypes(to_obj_type):
                # Use input relations in order to ensure connection order is preserved in cases where this matter
                # E.g. Junctions where tunnel_loss_1 might differ from tunnel_loss_2 and so on
                relations = shop.shop_api.GetInputRelations(to_obj_type, to_obj_name, conn_type)
                
                if not relations:
                    continue
                
                for r in relations:
                    new_relation = dict(obj_dicts[r])
                    new_relation["conn_type"] = conn_type
                    input_relations.append(new_relation)
                    
            if input_relations:
                # We also need to list the connections in downstream-upstream convention to preserve connection order
                for relation in input_relations:
                    if to_obj_type not in connections:
                        connections[to_obj_type] = {}
                    if to_obj_name in connections[to_obj_type]:
                        connections[to_obj_type][to_obj_name].append({'upstream_obj_type': relation['obj_type'],
                                                                      'upstream_obj_name': relation['obj_name'],
                                                                      'connection_type': relation['conn_type']})
                    else:
                        connections[to_obj_type][to_obj_name] = [{'upstream_obj_type': relation['obj_type'],
                                                                  'upstream_obj_name': relation['obj_name'],
                                                                  'connection_type': relation['conn_type']}]
                    
        return connections
    #    obj_list = list(
    #        zip(shop.shop_api.GetObjectTypesInSystem(), 
    #            shop.shop_api.GetObjectNamesInSystem())
    #    )
    #    obj_dicts = [{'obj_type': x[0], 'obj_name': x[1]} for x in obj_list]
    #    
    #    connections = {}
    #
    #    for to_obj_type, to_obj_name in obj_list:
    #        if shop.shop_api.GetObjectInfo(to_obj_type, 'isInput') == 'False':
    #            continue
#
    #        input_relations = []
    #        for conn_type in shop.shop_api.GetValidRelationTypes(to_obj_type):
    #            # Use input relations in order to ensure connection order is preserved in cases where this matter
    #            # E.g. Junctions where tunnel_loss_1 might differ from tunnel_loss_2 and so on
    #            relations = shop.shop_api.GetInputRelations(to_obj_type, to_obj_name, conn_type)
    #            
    #            if not relations:
    #                continue
    #            
    #            for r in relations:
    #                new_relation = dict(obj_dicts[r])
    #                new_relation["conn_type"] = conn_type
    #                input_relations.append(new_relation)
    #                
    #        if input_relations:
    #            # It is natural to list relation in from-to order, so we need to flip the retrieved input relations
    #            for relation in input_relations:
    #                if not relation['obj_type'] in connections:
    #                    connections[relation['obj_type']] = {}
    #                if relation['obj_name'] in connections[relation['obj_type']]:
    #                    connections[relation['obj_type']][relation['obj_name']].append([to_obj_type, to_obj_name,
    #                                                                                    relation['conn_type']])
    #                else:
    #                    connections[relation['obj_type']][relation['obj_name']] = [[to_obj_type, to_obj_name,
    #                                                                                relation['conn_type']]]
    #                
    #    return connections

    def _to_json_type(self, x):
        """ Convert value x into JSON type(s). """
             
        if isinstance(x, pd.Timestamp):
            return str(x)        

        if isinstance(x, np.float):
            return float(x)

        if isinstance(x, np.int):
            return int(x)
        
        if isinstance(x, pd.Series):
            if isinstance(x.index, pd.DatetimeIndex):
                s = x[x != x.shift(1)]  #  Drop concecutive identical values 
                return pd.Series(s.values, index=s.index.to_native_types()).to_dict()
            else:  # Cannot use to_dict directly as there may be duplicate index values (e.g. for endpoint_desc_nok_mm3)
                return {
                    'index': x.index.tolist(), 
                    'value': x.values.tolist(), 
                    'name': float(x.name) if isinstance(x.name, np.float64) else str(x.name)
                    }

        if isinstance(x, pd.DataFrame):
            df = x.loc[(x.shift() != x).all(1)]  # Drop consecutive identical rows
            df.index = df.index.to_native_types()

            # Get the longest string in the serialized data in order to adjust the cell size for visual alignment
            max_string_len = df.applymap(lambda y: len(str(y))).values.max()
            max_column_name_len = np.max([len(str(x)) for x in df.columns])
            cell_size = max(max_string_len, max_column_name_len)

            # Add a header with scenario indexing, for easier cross referencing
            data_dict = {f'{"Scenario #":>19}': ' '.join([f'{str(x):{">" + str(cell_size + 1)}}'
                                                          for x in df.columns])}
            # Convert each row of data, reflecting a single timestep, into a string with predefined cell width for each
            # value
            for row in df.itertuples(name=False):
                row_string = ''
                for value in row[1:]:
                    row_string += f'{str(value):{">" + str(cell_size + 1)}} '
                data_dict[row[0]] = row_string[:-1]
            return data_dict

        if isinstance(x, list):     
            return [self._to_json_type(i) for i in x]

        if x is None or isinstance(x, (int, float, str)):
            return x

        raise TypeError(f'<_to_json_type> : Unrecognized type {type(x)}')

    def _get_dict_with_json_types(self):

        s = self.copy()
        d = s.case
        
        for i in d['time']:
            d['time'][i] = s._to_json_type(d['time'][i])
            
        for obj_type, obj_name, attr, value in s._crawl_model():
            d['model'][obj_type][obj_name][attr] = s._to_json_type(value) 
            
        return s.case

    def _convert_to_pyshop_types(self):
        model = self.case['model']
        time = self.case['time']
        
        # Convert time data to types expected by pyshop
        if not isinstance(time['timeresolution'], (str, pd.Series)):
            time['timeresolution'] = pd.Series(
                    time['timeresolution']['value'], 
                    index=time['timeresolution']['index'], 
                    name=time['timeresolution']['name']
            )
        time['starttime'] = pd.Timestamp(time['starttime'])
        time['endtime'] = pd.Timestamp(time['endtime'])
        
        # Convert model data to pandas Series      
        for obj_type, obj_name, attr, value in self._crawl_model():
            if isinstance(value, dict):
                # Stochastic TXY
                if f'{"Scenario #":>19}' in value:
                    value_copy = dict(value)
                    value_copy.pop(f'{"Scenario #":>19}')
                    new_data = []
                    new_index = []
                    for k, v in value_copy.items():
                        new_index.append(k)
                        new_data.append([float(x) for x in v.strip().split()])
                    new_data = np.array(new_data)
                    new_value = pd.DataFrame(data=new_data, index=pd.to_datetime(new_index))
                    model[obj_type][obj_name][attr] = new_value.sort_index()
                # XY curve
                elif 'index' in value and 'value' in value:
                    model[obj_type][obj_name][attr] = pd.Series(value['value'], index=value['index'],
                                                                name=value['name'])
                # TXY
                else:
                    new_value = pd.Series(value)
                    new_value.index = pd.to_datetime(new_value.index)
                    model[obj_type][obj_name][attr] = new_value.sort_index()
            # XY Curve array
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                model[obj_type][obj_name][attr] = [pd.Series(i['value'], index=i['index'], name=i['name'])
                                                   for i in value]
