
from __future__ import annotations

from typing import ByteString, Union, Dict, Iterable

import os
import json
import zipfile
import tempfile
import time
import logging
import uuid
from copy import deepcopy
from pathlib import Path
from collections import namedtuple
from io import BytesIO

import pandas as pd
import numpy as np
import yaml
from graphviz import Digraph

import pyshop  # type: ignore


log = logging.getLogger(__name__)


class DictImitator:
    """ To enable .<tab> completion of ShopCase.case in Jupyter. 
    
    Methods "__getitem__", "keys" and "__contains__" are required to make the class
    play nice with pandas. E.g. to make the pandas.Series initiation from DictImitator 
    behave in the same way as for native dicts.

    Known limitations
    -----------------
    pandas.DataFrame(DictImitator) != pandas.DataFrame(DictImitator.to_dict())
        Use pandas.DataFrame(DictImitator).to_dict()
        The reason is that pandas checks for isinstance(..., dict) when initializing the
        arrays in the DataFrame. (suggest that pandas checks is_dict_like instead from 
        \\pandas\\core\\dtypes\\inference.py instead?).
        A solution could be to inherit from dict, but this can create unpredictable 
        behaviour and is not recommended 
        https://treyhunner.com/2019/04/why-you-shouldnt-inherit-from-list-and-dict-in-python/
        Adding tests of all overridden methods may allow us to inherit from dict in a 
        safe way.
    """
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                self.__dict__[key] = DictImitator(**value)
                
    def __repr__(self):
        return self.__dict__.__repr__()
    
    def __contains__(self, key):
        return key in self.__dict__      
    
    def __getitem__(self, key):
        return self.__getattribute__(key)
        
    def __setitem__(self, key, value):
        self.__setattr__(key, value)
        
    def __setattr__(self, key, value):
        self.__dict__[key] = DictImitator(**value) if isinstance(value, dict) else value
        
    def __iter__(self):
        return self.__dict__.__iter__()

    def __len__(self):
        return len(self.__dict__)
    
    def keys(self):
        return self.__dict__.keys()
    
    def items(self):
        return self.__dict__.items()
    
    def values(self):
        return self.__dict__.values()

    def update(self, d):
        self.__dict__.update(d)

    def pop(self, key):
        return self.__dict__.pop(key)
    
    def to_dict(self):
        d = {}
        for k, v in self.items():
            if isinstance(v, self.__class__):
                v = v.to_dict()
            d[k] = v
        return d


class ShopCaseBaseClass:
    
    def __init__(self, source, shop_init_func=pyshop.ShopSession):

        log.info(f'Init ShopCase : source_type={type(source)}')

        self.shop_init_func = shop_init_func
        self.case = None

        if isinstance(source, pyshop.ShopSession):
            self._from_shopsession(source)
        elif isinstance(source, (dict, DictImitator)):
            self.case = source
        elif isinstance(source, bytes):
            self._from_bytestring(source)
        elif isinstance(source, str) and source[0] == '{':
            self._from_json(source)
        #elif isinstance(source, 'yaml_string'):
        #    raise NotImplementedError
        elif isinstance(source, (str, Path)) and (Path(source) / 'model.yaml').exists():
            self._from_dir(source)
        elif isinstance(source, (str, Path)) and '.shop.zip' in Path(source).name:
            self._from_file(source)

        if isinstance(self.case, dict):
            self.case = DictImitator(**self.case)
        
    @property
    def model(self):
        return self.case['model']

    @property
    def time(self):
        return self.case['time']

    @property
    def commands(self):
        return self.case['commands']

    @property
    def connections(self):
        return self.case['connections']

    @property
    def ascii(self):
        return self.case['ascii']

    def run(
        self, 
        shopdir=None,
        save_results_per_iteration: bool = False
        ) -> pyshop.ShopSession:
        """ Run the ShopCase in pyshop, and update self with the new results. """

        results_per_iteration = []

        # Move to a temporary directory
        _cwd = os.getcwd()
        if shopdir is None:
            shopdir = tempfile.TemporaryDirectory(prefix='shopcase_').name
            os.mkdir(shopdir)
            os.chdir(shopdir)
        else: 
            shopdir = _cwd()
        self.shopdir = shopdir

        # Run SHOP
        iteration = 0
        shop = self.to_shopsession()
        for c in self.case['commands']:
            if not c.strip() or c.strip()[0] == '#':
                continue
            time.sleep(0.1)  # To prevent delayed print of command in Jupyter
            if 'start sim' in c:
                for i in range(int(c.split(' ')[-1])):
                    log.info(f'Calling command "start sim 1"')
                    shop.execute_full_command('start sim 1') 
                    if save_results_per_iteration:
                        iteration += 1
                        results_per_iteration.append(
                            ShopCaseBaseClass(shop)
                            )
            else:
                log.info(f'Calling command "{c}"')
                shop.execute_full_command(c) 

        # Reverse ownership scaling
        # https://shop.sintef.energy/discussions/shop-api-pyshop/re-running-cases-when-ownership-is-not-100/
        # Don't use as owmnership will be set to 100 % for all plants, waiting for a change
        # Nils Ræder, 2021-02-17
        #shop.execute_full_command('reset ownership')

        # Preserve custom fields in self.case
        _old_case = self.case 
        self._from_shopsession(shop)
        for i in _old_case:
            if i not in self.case:
                self.case[i] = _old_case[i]

        self.case = DictImitator(**self.case)

        # Save the logs
        self.case['logs'] = {}
        self.case['logs']['shop'] = shop.get_messages()
        if Path('cplex.log').exists():
            self.case['logs']['cplex'] = Path('cplex.log').read_text()

        # Return to previous working directory
        os.chdir(_cwd)

        if results_per_iteration:
            return results_per_iteration

        return shop

    def open_shopdir(self):
        if hasattr(self, 'shopdir'):
            os.startfile(self.shopdir)      

    def copy(self):      
        """ Create a new ShopCase instance from a deep copy of self.case."""
        return self.__class__(deepcopy(self.case))

    def to_shopsession(self) -> pyshop.ShopSession:

        # Fix for autogenerated reservoir.start_vol
        # https://shop.sintef.energy/discussions/shop-api-pyshop/auto-generated-start_vol-causes-problems/
        for r in self.case['model']['reservoir'].values():
            if ('start_vol' in r) and ('start_head' in r) and (r['start_head'] > 0):
                r.pop('start_vol')

        shop = self.shop_init_func()
        
        commands_to_call_before_object_creation = ['set newgate /off']
        for c in commands_to_call_before_object_creation:
            if c in self.case['commands']:
                log.info(f'Calling command "{c}"')
                shop.execute_full_command(c)

        model = self.case['model']

        # Set time resolution
        if 'time' in self.case:
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
            
        # Insert connections in ShopSession
        sorted_connections = sorted(
            self.case['connections'], key=lambda c: c.get('order', 0)
            )
        for c in sorted_connections:
            shop.shop_api.AddRelation(
                c['from_type'], c['from'], c['connection_type'], c['to_type'], c['to']
                )

        # Load any ascii data
        if 'ascii' in self.case:
            for i in self.case['ascii']:
                log.info(f'Loading ASCII string {i["name"]}')
                shop.shop_api.ReadShopAsciiString(i['content'])

        return shop

    def describe(self):
        print('### Objects in model ###')
        indent = ' '
        for k, v in self.case['model'].items():
            print(k)
            if not isinstance(v, (dict, DictImitator)):
                continue
            for k2, v2 in v.items():
                print(indent, k2)          

    def diff(self, other: 'ShopCaseBaseClass') -> Dict:
        """ Compare the data in two ShopCases and return a dict with the same 
        structure as ShopCase.case indicating where the data is different.

        Returns an empty dict if the cases are identical (within the tolerance).
        
        The tolerance applies to data in pandas.Series, and is by default set
        to 0.1, and should be less than 1 to catch differences in binary values
        like unit commitment.
        """

        def _are_equal_despite_different_types(i, j):
            if isinstance(i, pd.Series) and (len(set(i.round(5))) == 1):
                i = i.iloc[0]
            if isinstance(j, pd.Series) and (len(set(j.round(5))) == 1):
                j = j.iloc[0]
            if isinstance(i, pd.Series) or isinstance(j, pd.Series):
                return False
            return i == j

        def _series_are_equal(i: pd.Series, j: pd.Series):
            if not type(i.index) is type(j.index):
                return False
            if i.index.has_duplicates or j.index.has_duplicates:
                return i.round(4).equals(i.round(4))
            common_index = i.index | j.index
            i = i.reindex(common_index).ffill()
            j = j.reindex(common_index).ffill()
            return (i - j).abs().max() < 0.0001

        def _compare(left, right):
            """ Compare two dictionaries. """
            
            diffs = {}
            
            for i in set(left) | set(right):
                if i not in right or i not in left:
                    diffs[i] = True
                elif isinstance(left[i], (dict, DictImitator)):
                    res = _compare(left[i], right[i])
                    if res:
                        diffs[i] = res
                else:
                    status = _is_different(left[i], right[i])
                    if status:
                        diffs[i] = status
                        
            return diffs
       
        def _is_different(i, j):
            """ Compare two objects. """

            if not type(i) is type(j):
                return not _are_equal_despite_different_types(i, j)
            if isinstance(i, (int, float, str, pd.Timestamp)):
                return i != j
            if isinstance(i, pd.Series):
                return not _series_are_equal(i, j)
            if isinstance(i, (dict, DictImitator)):
                return _compare(i, j)
            if isinstance(i, list):
                if len(i) != len(j):
                    return True
                return any([_is_different(m, n) for m, n in zip(i, j)])
            
            raise ValueError(f'Type not specified in is_equal : {i}={type(i)}')

        return _compare(self.case, other.case)

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

    def expand_timeseries(self, freq: str = '15T'):
        """ Expand all time series to include all time steps between starttime and 
        endtime with the given frequency.

        Note that this may distort the aggregate sum of e.g. startup costs.        
        """
        for obj_type, obj_name, attr, value in self._crawl_model():
            if isinstance(value, pd.Series) and isinstance(value.index, pd.DatetimeIndex):
                value.loc[self.case['time']['endtime']] = value.iloc[-1]
                self.case['model'][obj_type][obj_name][attr] = value.resample(freq).ffill()

    def show_topology(self, return_img=False, img_width='100%') -> Digraph:
        """ 
        Show the model topology using graphviz. Differs from the build_connection_tree
        method in pyshop by including *all* objects.

        parameters
        ----------
        return_img : bool
            Return an HTML <img> tag with a PNG image. Necessary for scaling the topology to
            fit the width of a jupyter notebook and to preserve the image when the 
            notebook is saved. Only works in an IPython environment.
        img_width : str
            The with of the image set in the <img> tag, e.g. '100%'.
        """

        if return_img:
            from IPython.display import HTML
            import base64
            top = self.show_topology()
            top.format = 'png'
            return HTML(f'<img width={img_width} src="data:image/png;base64,{base64.b64encode(top.pipe()).decode()}">')

        def get_edge_style(e: Edge):
            if 'gate' in [e.start.type, e.end.type] and e.relation == 'connection_standard':
                return 'dashed'
            return 'solid'  

        # Styles
        default_node_attrs = {'shape': 'ellipse', 'fillcolor': '#ffffff', 'style': ''}
        node_attrs = {
            'plant': {'shape': 'box', 'fillcolor': '#FF9999', 'style': 'filled'},
            'reservoir': {'shape': 'invtriangle', 'fillcolor': '#99FFFF', 'style': 'filled'},
            'junction': {'shape': 'point', 'fillcolor': '#ffffff', 'style': ''},
            'junction_gate': {'shape': 'point', 'fillcolor': '#ffffff', 'style': ''}
        }

        dot = Digraph(comment='SHOP topology')
        Node = namedtuple('Node', ['type', 'label', 'name'])
        Edge = namedtuple('Edge', ['relation', 'start', 'end'])
        nodes = set()
        edges = set()

        for c in self.case['connections']:
            node1 = Node(type=c['from_type'], label=c['from'], name=f"{c['from_type']}_{c['from']}")
            node2 = Node(type=c['to_type'], label=c['to'], name=f"{c['to_type']}_{c['to']}")
            edge = Edge(relation=c['connection_type'], start=node1, end=node2)         
            nodes.add(node1)
            nodes.add(node2)
            edges.add(edge)

        for n in nodes:
            dot.node(n.name, label=n.label, **node_attrs.get(n.type, default_node_attrs))

        for e in edges:
            dot.edge(e.start.name, e.end.name, style=get_edge_style(e))

        return dot

    def to_file(self, filename: Union[str, Path] = 'case.shop.zip'):
        # Drop bestprofit and other marginal cost data as it inflates the model 
        # file to an unmanagable size.
        self.drop_mc_data()  
        with open(filename, 'wb') as f:
            f.write(self._to_bytestring())
        return Path(filename).absolute()

    def _to_bytestring(self) -> ByteString:
        b = BytesIO()
        with zipfile.ZipFile(b, 'w', zipfile.ZIP_DEFLATED) as f:
            for key, value in self._get_dict_with_json_types().items():
                f.writestr(f'{key}.yaml', yaml.dump(value, allow_unicode=True, encoding='UTF-8', sort_keys=False)) #.encode('utf-8')
        return b.getvalue()

    def to_json(self, indent=4):
        return json.dumps(
            self._get_dict_with_json_types(), 
            ensure_ascii=False, 
            indent=indent
            )

    def to_yaml_string(
        self, subset: Iterable[str] = None, drop_custom_fields: bool = True
        ) -> str:
        d = self._get_dict_with_json_types()
        if subset is not None:
            [d.pop(i) for i in list(d) if not i in subset]
        elif drop_custom_fields:
            [d.pop(i) for i in list(d) if i[0] == '_']
        return yaml.safe_dump(
            d,
            allow_unicode=True, 
            encoding='UTF-8', 
            sort_keys=False
            ).decode()
        
    def to_yaml_files(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        if not path.exists():
            path.mkdir()
        for key, value in self._get_dict_with_json_types().items():
            with open(path / f'{key}.yaml', 'bw') as f:
                f.write(yaml.safe_dump(value, allow_unicode=True, encoding='UTF-8', sort_keys=False)) #.encode('utf-8')
        return path.absolute()

    def _from_json(self, s: str):
        self.case = json.loads(s)
        self._convert_to_pyshop_types()

    def _from_yaml_string(self, s: str):
        raise NotImplementedError

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
        
    def _from_shopsession(self, shop: pyshop.ShopSession):
        fields = {
            'model': {'func': self._get_model, 'default': {}},
            'connections': {'func': self._get_connections, 'default': {}},
            'time': {'func': self._get_time, 'default': {}},
            'commands': {'func': self._get_commands, 'default': []}
            }
        d = {}
        for k, v in fields.items():
            try: 
                value = v['func'](shop)
            except Exception as e:
                log.error(f"Couldn't import {k} from ShopSession due to '{e}'")
                value = v['default']
            d[k] = value

        # Necessary? To avoid any reference to the same instances in ShopCase 
        # and ShopSession
        self.case = deepcopy(d)  

        # Drop identical consecutive values in timeresolution
        x = self.case['time']['timeresolution']
        self.case['time']['timeresolution'] = x[x != x.shift(1)]

        # Fix for autogenerated reservoir.start_vol
        # https://shop.sintef.energy/discussions/shop-api-pyshop/auto-generated-start_vol-causes-problems/
        for r in self.case['model']['reservoir'].values():
            if ('start_vol' in r) and ('start_head' in r) and (r['start_head'] > 0):
                r.pop('start_vol')

    def _get_time(self, shop):
        return shop.get_time_resolution() 

    def _get_commands(self, shop):
        return shop.get_executed_commands()

    def _get_model(self, shop: pyshop.ShopSession) -> Dict:
        """ Dump all data in ShopSession.model to nested dict. """
        
        obj_list = list(
            zip(shop.shop_api.GetObjectTypesInSystem(), 
                shop.shop_api.GetObjectNamesInSystem())
        )
        
        model = dict()
        
        for obj_type, obj_name in obj_list:   
            try: 
                obj = shop.model[obj_type][obj_name]
            except AttributeError:
                log.error(f'_get_model : Could not get shop.model[{obj_type}][{obj_name}]')
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
            
    def _get_connections(self, shop: pyshop.ShopSession) -> Dict:
        connections = []
        obj_types = shop.shop_api.GetObjectTypesInSystem()
        obj_names = shop.shop_api.GetObjectNamesInSystem()

        for to_type, to_name in zip(obj_types, obj_names):
            valid_relation_types = shop.shop_api.GetValidRelationTypes(to_type)
            
            for conn_type in valid_relation_types:
                relations = shop.shop_api.GetInputRelations(to_type, to_name, conn_type)
                
                for order, r in enumerate(relations): # Enumerate to get "order"
                    from_type, from_name = obj_types[r], obj_names[r]
                    connection = {
                        'from': from_name, 
                        'to': to_name,
                        'from_type': from_type,
                        'to_type': to_type,
                        'connection_type': conn_type
                    }
                    if len(relations) > 1 and 'junction' in to_type:
                        connection['order'] = order
                    connections.append(connection)

        return connections

    def _to_json_type(self, x):
        """ Convert value x into JSON type(s). """

        if isinstance(x, (dict, DictImitator)):
            return dict([(k, self._to_json_type(v)) for k, v in x.items()])
             
        if isinstance(x, pd.Timestamp):
            return str(x)        

        if isinstance(x, np.float):
            return float(x)

        if isinstance(x, np.int):
            return int(x)
        
        if isinstance(x, pd.Series) and isinstance(x.index, pd.DatetimeIndex):
            s = x[x != x.shift(1)]  #  Drop concecutive identical values 
            return pd.Series(s.values, index=s.index.to_native_types()).to_dict()

        if isinstance(x, pd.Series):  # Will remove duplicate index values (e.g. for endpoint_desc_nok_mm3)
            return {
                'x': [self._to_json_type(i) for i in x.index], 
                'y': x.values.tolist(), 
                'ref': self._to_json_type(x.name)
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
        d = deepcopy(self.case.to_dict())  # Necessary to deepcopy?
        return self._to_json_type(d)

    def _convert_to_pyshop_types(self):

        if 'time' in self.case:
            time = self.case['time']
            
            # Convert time data to types expected by pyshop
            tr = time['timeresolution']
            if isinstance(tr, (dict, DictImitator)):
                # Time resolution is a XY curve
                if 'x' in tr:
                    time['timeresolution'] = pd.Series(tr['y'], index=tr['x'], name=tr['ref'])
                # Time resolution is a TXY curve
                else:
                    time['timeresolution'] = pd.Series(tr)
                    time['timeresolution'].index = pd.to_datetime(time['timeresolution'].index)

            # Drop identical consecutive values in timeresolution
            time['timeresolution'] = time['timeresolution'] \
                [time['timeresolution'] != time['timeresolution'].shift(1)]
                
            time['starttime'] = pd.Timestamp(time['starttime'])
            time['endtime'] = pd.Timestamp(time['endtime'])
        

        # Convert model data to pandas Series      
        model = self.case['model']
        for obj_type, obj_name, attr, value in self._crawl_model():
            if isinstance(value, (dict, DictImitator)):
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
                elif 'x' in value and 'y' in value:
                    new_value = pd.Series(
                        value['y'], 
                        index=value['x'],
                        name=value['ref']
                        )
                    model[obj_type][obj_name][attr] = new_value
                # TXY
                else:
                    new_value = pd.Series(value)
                    new_value.index = pd.to_datetime(new_value.index)
                    model[obj_type][obj_name][attr] = new_value.sort_index()
            # XY Curve array
            elif isinstance(value, list) \
                and all([isinstance(i, dict) for i in value]) \
                and all([all([j in i for j in ['x', 'y', 'ref']]) for i in value]):
                model[obj_type][obj_name][attr] = \
                    [pd.Series(i['y'], index=i['x'], name=i['ref']) for i in value]
