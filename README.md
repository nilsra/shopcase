
| :warning: This repository is NOT part of the SHOP software, or supported by Sintef in any way. It purely a demonstration of how an alternative data structure using open standards can be implemented. |
|---|

## Purpose of this repository
To facilitate cooperation between users and developers of the SHOP
optimization software.

## What this is
This repository is intended as a reference implementation for YAML SHOP, 
demonstrating some of the benefits of having a serializable data structure 
based on open standards. 

The open standard in this case is the JSON compatible subset of YAML. 

## Functionality
```ShopCaseBaseClass``` includes functionality for converting a set of SHOP case data 
to and from ```pyshop.ShopSession``` instances, YAML files, zip files and 
JSON strings. It also includes functionality for copying and comparing SHOP data, and 
running SHOP cases in an standardized way while writing files to directories 
under ```%TEMP%```.

## Examples
See the notebook under ```/doc```.
