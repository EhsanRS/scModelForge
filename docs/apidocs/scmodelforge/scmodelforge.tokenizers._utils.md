---
orphan: true
---

# {py:mod}`scmodelforge.tokenizers._utils`

```{py:module} scmodelforge.tokenizers._utils
```

```{autodoc2-docstring} scmodelforge.tokenizers._utils
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ensure_tensor <scmodelforge.tokenizers._utils.ensure_tensor>`
  - ```{autodoc2-docstring} scmodelforge.tokenizers._utils.ensure_tensor
    :summary:
    ```
* - {py:obj}`rank_genes_by_expression <scmodelforge.tokenizers._utils.rank_genes_by_expression>`
  - ```{autodoc2-docstring} scmodelforge.tokenizers._utils.rank_genes_by_expression
    :summary:
    ```
* - {py:obj}`compute_bin_edges <scmodelforge.tokenizers._utils.compute_bin_edges>`
  - ```{autodoc2-docstring} scmodelforge.tokenizers._utils.compute_bin_edges
    :summary:
    ```
* - {py:obj}`digitize_expression <scmodelforge.tokenizers._utils.digitize_expression>`
  - ```{autodoc2-docstring} scmodelforge.tokenizers._utils.digitize_expression
    :summary:
    ```
````

### API

````{py:function} ensure_tensor(x: numpy.ndarray | torch.Tensor, dtype: torch.dtype) -> torch.Tensor
:canonical: scmodelforge.tokenizers._utils.ensure_tensor

```{autodoc2-docstring} scmodelforge.tokenizers._utils.ensure_tensor
```
````

````{py:function} rank_genes_by_expression(expression: torch.Tensor, gene_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
:canonical: scmodelforge.tokenizers._utils.rank_genes_by_expression

```{autodoc2-docstring} scmodelforge.tokenizers._utils.rank_genes_by_expression
```
````

````{py:function} compute_bin_edges(values: numpy.ndarray | None = None, n_bins: int = 51, method: str = 'uniform', value_max: float = 10.0) -> numpy.ndarray
:canonical: scmodelforge.tokenizers._utils.compute_bin_edges

```{autodoc2-docstring} scmodelforge.tokenizers._utils.compute_bin_edges
```
````

````{py:function} digitize_expression(values: torch.Tensor, bin_edges: numpy.ndarray) -> torch.Tensor
:canonical: scmodelforge.tokenizers._utils.digitize_expression

```{autodoc2-docstring} scmodelforge.tokenizers._utils.digitize_expression
```
````
