Field Fitter
============

Scaffold/model field fitting library using OpenCMISS-Zinc.

Takes as input a Zinc model file and a Zinc data file.
Model requires a model coordinates field, and data a data coordinates field
which are in the same aligned/fitted configuration.

Additional fields defined on the data may be defined and fitted to a continuous field
on the model or a subgroup of it. The fitted field is defined using the
basis and mappings of the first component of the model coordinates field.

