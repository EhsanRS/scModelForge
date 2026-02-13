# {py:mod}`scmodelforge.data.census`

```{py:module} scmodelforge.data.census
```

```{autodoc2-docstring} scmodelforge.data.census
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`build_obs_value_filter <scmodelforge.data.census.build_obs_value_filter>`
  - ```{autodoc2-docstring} scmodelforge.data.census.build_obs_value_filter
    :summary:
    ```
* - {py:obj}`load_census_adata <scmodelforge.data.census.load_census_adata>`
  - ```{autodoc2-docstring} scmodelforge.data.census.load_census_adata
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <scmodelforge.data.census.logger>`
  - ```{autodoc2-docstring} scmodelforge.data.census.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: scmodelforge.data.census.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} scmodelforge.data.census.logger
```

````

````{py:function} build_obs_value_filter(filters: dict[str, typing.Any]) -> str | None
:canonical: scmodelforge.data.census.build_obs_value_filter

```{autodoc2-docstring} scmodelforge.data.census.build_obs_value_filter
```
````

````{py:function} load_census_adata(census_config: scmodelforge.config.schema.CensusConfig, obs_keys: list[str] | None = None) -> anndata.AnnData
:canonical: scmodelforge.data.census.load_census_adata

```{autodoc2-docstring} scmodelforge.data.census.load_census_adata
```
````
