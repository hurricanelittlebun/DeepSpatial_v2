# Morphology-aware DeepSpatial implementation plan

Approved specification: the user's Morphology-aware UOT + Morphology-guided
Spatial Flow design. No registration changes, no expression-to-UNI2 gradients.

1. Add physical, chunked HDF5 UNI2 grid storage, XY bilinear/Z linear queries,
   frozen UNI2-h encoder and offline physical-FOV extraction. Test coordinate
   interpolation with a hand-computed affine feature field and invalid bounds.
2. Extend UOT optional costs, keeping original defaults exactly. Test cost and
   actual Sinkhorn coupling with matching versus exchanged morphology features.
3. Cache local-candidate layered DP paths as PCHIP coefficients keyed by endpoints,
   feature revision and parameters; evaluate in torch at random t. Test curved
   corridor, endpoints and finite-difference dx/dt on nonuniform physical Z.
4. Wire dataset pair IDs, path targets and physical feature queries. Preserve
   cell-type states and add explicit no-celltype mode. Keep H&E out of the GiT
   input heads; test the three public modes through the real API.
5. Wire forward/reverse ODE and checkpoint metadata, preserving original-mode
   numerics. Test reconstruction and checkpoint roundtrip. Document units, cache
   invalidation, flags, performance limits, real-checkpoint prerequisites.

Implementation boundaries: histology/config.py, feature_store.py, uni2.py,
preprocessing.py, path.py, runtime.py; targeted changes to core.py, dataset.py,
uot_solver.py and module.py. Generic transport remains unchanged. The released
GiT keeps its original spatial/gene/cell-type heads and does not expose an H&E
late-fusion spatial head.
One feature store belongs to ONE independently registered tissue/g series.
All morphology costs/search operate in micrometres; training remains normalized.
