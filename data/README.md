This folder contains a JSON-file with an example annotation. 
More examples with `.nii` files will appear soon.

## Annotations format

Each file must contain a list of annotations.
Each annotation is a list of curves, so that `curve[i]` is the annotation for the i-th axial slice.
Each curve is a list of 2D points (y, x) that define the curve.
If a given slice doesn't have an annotation - the curve is an empty list.

E.g.:

```
[
  [ // first annotation
    [], // empty slice
    [], // empty slice
    [[50.2, 210.12], [55.4, 215.23], [60.1, 213.54], ...],
    ...
    [[57.4, 214.2], [66.2, 209.53], ...],
    [], // empty slice
    []  // empty slice
  ],
  [ // second annotation
    ...
  ],
  ...
]
```
