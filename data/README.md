# data/

Local working data. **Contents are gitignored** — this repo does not
redistribute LifeSphere's underlying data. Only this README is tracked.

| Directory    | Holds                                                                 |
| ------------ | --------------------------------------------------------------------- |
| `raw/`       | Verbatim pulls from Neo4j, unmodified (strings still strings).         |
| `interim/`   | Typed and joined frames — casts applied, missingness made explicit.    |
| `processed/` | Model-ready artefacts, one subdirectory per graph construction or model. |

Everything here is reproducible from the live graph via `gl_lifesphere.extract`;
nothing here is a source of truth.
