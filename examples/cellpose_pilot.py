"""Reproducible section-3 ROI pilot; run extract and segment in their own environments."""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['extract', 'segment', 'register'])
    parser.add_argument('--output', default='data/00029_g0/cellpose_pilot/section-003')
    parser.add_argument(
        '--registration-root',
        type=Path,
        help='external saved-registration directory, required for register',
    )
    parser.add_argument(
        '--cellpose-model',
        type=Path,
        help='local Cellpose checkpoint, required for segment',
    )
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler(out/'run.log')])
    if args.stage == 'register':
        import pandas as pd
        from deepspatial.histology.registered_coordinates import source_to_registered_points
        from deepspatial_v2.data_export.cell_level_alignment import SavedTransformStore, load_section_transform
        meta = json.loads((out/'metadata.json').read_text())
        table = pd.read_csv(out/'nuclei.csv')
        if args.registration_root is None:
            parser.error('--registration-root is required for register')
        store = SavedTransformStore(args.registration_root, device='cpu')
        section = load_section_transform(meta['transform_path'])
        xy = source_to_registered_points(table[['x_source_um','y_source_um']].to_numpy(), section, store)
        table['x_registered_um'], table['y_registered_um'] = xy.T
        table['coordinate_frame'] = '00029__g0_registered'
        table.to_csv(out/'nuclei_registered.csv', index=False)
        logging.info('Mapped %d nuclei through saved registration', len(table))
        return
    if args.stage == 'extract':
        import pandas as pd
        from scipy.ndimage import distance_transform_edt
        import importlib.util
        spec = importlib.util.spec_from_file_location('pilot_sdpc', Path(__file__).resolve().parents[1]/'deepspatial/histology/sdpc.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        SdpcPyramid = module.SdpcPyramid
        row = pd.read_parquet('data/00029_g0/he_sections.parquet').query('section_id == 3').iloc[0]
        mask = np.asarray(Image.open(row.source_mask_path).convert('L')) > 0
        y, x = np.unravel_index(distance_transform_edt(mask).argmax(), mask.shape)
        center_um = (np.array([x, y]) + [row.bbox_x0, row.bbox_y0]) * [row.analysis_pixel_size_x_um, row.analysis_pixel_size_y_um]
        with SdpcPyramid(row.raw_source_path) as reader:
            mpps = np.asarray(reader.level_downsamples) * reader.mpp_level0
            level = int(np.argmin(abs(np.log(mpps / 0.5))))
            size = 1024
            origin = np.rint(center_um / reader.mpp_level0 - size * reader.level_downsamples[level] / 2).astype(int)
            rgb = reader.read_region(tuple(origin), level, (size, size))
            meta = dict(section_id='3', z_um=float(row.z_um), source_path=row.raw_source_path,
                        origin_level0=origin.tolist(), mpp_level0=reader.mpp_level0,
                        mpp=float(mpps[level]), level=level, size=size,
                        transform_path=row.section_transform_path, scope='1024-pixel tissue-interior ROI only')
        Image.fromarray(rgb).save(out/'he_roi.png')
        (out/'metadata.json').write_text(json.dumps(meta, indent=2))
        logging.info('Extracted ROI: %s', meta)
        return

    import torch
    import tifffile
    import pandas as pd
    from cellpose import models
    from skimage.color import rgb2hed
    from skimage.measure import regionprops
    from skimage.segmentation import find_boundaries
    from importlib.metadata import version
    rgb = np.asarray(Image.open(out/'he_roi.png').convert('RGB'))
    meta = json.loads((out/'metadata.json').read_text())
    # Hematoxylin optical density: positive bright nuclei for the nuclei model.
    hematoxylin = np.maximum(rgb2hed(rgb)[..., 0], 0).astype('float32')
    low, high = np.percentile(hematoxylin, [1, 99])
    stain = np.clip((hematoxylin-low)/max(high-low, 1e-8), 0, 1)
    if args.cellpose_model is None:
        parser.error('--cellpose-model is required for segment')
    weights = str(args.cellpose_model)
    model = models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model=weights)
    logging.info('Cellpose %s; GPU=%s', version('cellpose'), torch.cuda.is_available())
    masks, flows, _ = model.eval(stain, diameter=8.0/meta['mpp'], channels=[0, 0],
                                normalize=False, cellprob_threshold=0, flow_threshold=0.4)
    tifffile.imwrite(out/'nuclei.tif', masks.astype('uint32'))
    np.save(out/'cellprob_score.npy', flows[2])
    rows=[]
    for region in regionprops(masks):
        y,x = region.centroid
        source = np.asarray(meta['origin_level0'])*meta['mpp_level0'] + np.array([x,y])*meta['mpp']
        rows.append(dict(cell_id=f'00029_g0_3_roi_n{region.label}', nucleus_id=region.label,
                         x_roi_px=x, y_roi_px=y, x_source_um=source[0], y_source_um=source[1],
                         z_um=meta['z_um'], area_um2=region.area*meta['mpp']**2,
                         perimeter_um=region.perimeter*meta['mpp'], eccentricity=region.eccentricity,
                         touches_roi_edge=region.bbox[0]==0 or region.bbox[1]==0 or region.bbox[2]==masks.shape[0] or region.bbox[3]==masks.shape[1]))
    pd.DataFrame(rows).to_csv(out/'nuclei.csv', index=False)
    overlay=rgb.copy()
    overlay[find_boundaries(masks, mode='inner')]=[0,255,0]
    panel=np.concatenate([rgb,overlay],axis=1)
    Image.fromarray(panel).save(out/'qc_original_and_boundaries.png')
    summary=dict(n_nuclei=len(rows), cellpose_version=version('cellpose'), model=weights,
                 diameter_um=8.0, score_semantics='raw Cellpose score, not calibrated confidence',
                 scope=meta['scope'], coordinate_frame='source slide micrometers; registration pending')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    logging.info('Finished: %s', summary)


if __name__=='__main__':
    main()
