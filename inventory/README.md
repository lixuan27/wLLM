# inventory/

Machine-generated catalog inventories live here and are **not tracked**:
they embed absolute paths of a local model-substrate checkout.

Regenerate against your local substrate checkout:

```bash
export WLLM_SUBSTRATE_ROOT=/path/to/substrate-checkout
python - <<'EOF'
from wllm.backends.catalog.importer import CatalogImporter
import os
CatalogImporter(os.environ["WLLM_SUBSTRATE_ROOT"]).save_inventory(
    "inventory/catalog_inventory.json")
EOF
```

The inventory records, per manifest: normalized id/category/tasks,
integration-status *claims* (never trusted as runnable), checkpoint
repos with license/gated flags, and schema-path hints for auditability.
