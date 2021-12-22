# Install without conda

This is a rough WIP procedure to try to get an install without conda:

```bash
sudo bash scripts/install-freecad-deps.sh
bash scripts/install-freecad.sh
```

It currently fails when trying to link against PySide2, so there's probably something
still not quite right with some versioning.
