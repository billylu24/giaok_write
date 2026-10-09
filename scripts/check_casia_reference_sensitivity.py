"""Post-run diagnostic: keep target/noise fixed and change only writer reference."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'vendor/FontDiffuser'))
from configs.fontdiffuser import get_parser
from src import (FontDiffuserModelDPM, FontDiffuserDPMPipeline, build_unet,
                 build_style_encoder, build_content_encoder, build_ddpm_scheduler)
from handwriting.casia import CASIADataset


def pairwise_mae(images):
    return float(np.mean([np.abs(images[i].astype(float) - images[j].astype(float)).mean() / 255
                          for i in range(len(images)) for j in range(i + 1, len(images))]))


def main():
    out = ROOT / 'experiments/casia_001/reference_sensitivity'
    if (out / 'metrics.json').exists():
        raise FileExistsError('Reference sensitivity results already exist')
    out.mkdir(parents=True, exist_ok=True)
    cases = json.loads((out.parent / 'evaluation_cases.json').read_text())['cases']
    targets = ['湖', '海', '永']
    cases = [c for c in cases if c['kind'] == 'gallery_characters' and c['target'] in targets]
    writers = list(dict.fromkeys(c['writer'] for c in cases))
    dataset = CASIADataset(ROOT / 'data/casia/processed', 'test', 123)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args = get_parser().parse_args([])
    args.style_image_size = args.content_image_size = (96, 96)
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 18, index=2)
    board = Image.new('RGB', (160 + 144 * 3, 44 + 176 * 6), '#edf2f7')
    draw = ImageDraw.Draw(board)
    for i, char in enumerate(targets):
        draw.text((200 + i * 144, 10), char, font=font, fill='black')
    results = {}
    for pi, (phase, checkpoint) in enumerate([('before', 'checkpoints/fontdiffuser'),
                                             ('after', 'experiments/casia_001/checkpoints/best')]):
        modules = {}
        for name, builder in [('unet', build_unet), ('style_encoder', build_style_encoder),
                              ('content_encoder', build_content_encoder)]:
            model = builder(args)
            model.load_state_dict(torch.load(ROOT / checkpoint / f'{name}.pth', map_location='cpu', weights_only=True), strict=True)
            modules[name] = model
        model = FontDiffuserModelDPM(**modules).to('cuda').eval()
        pipeline = FontDiffuserDPMPipeline(model, build_ddpm_scheduler(args), guidance_scale=7.5)
        generated, references = {}, {}
        for case in cases:
            sample = dataset[case['dataset_index']]
            images, codes = dataset.writer_arrays(case['writer'])
            ref = images[case['reference_index']]
            assert chr(int(codes[case['reference_index']])) == '人'
            reference = dataset.tensor(ref).unsqueeze(0).to('cuda')
            seed = 30000 + targets.index(case['target'])
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                image = pipeline.generate(sample['content'].unsqueeze(0).to('cuda'), reference,
                                          1, 2, 20, args.content_encoder_downsample_size,
                                          dm_size=(96, 96), skip_type='time_uniform')[0]
            name = f"{phase}_{case['writer']}_{ord(case['target'])}.png"
            image.save(out / name)
            generated.setdefault(case['target'], []).append(np.asarray(image))
            references.setdefault(case['target'], []).append(np.asarray(ref))
            row = writers.index(case['writer']) * 2 + pi
            x = 164 + targets.index(case['target']) * 144
            y = 44 + row * 176
            draw.text((4, y + 40), f"{case['writer']} / {phase}", font=font, fill='black')
            board.paste(image.resize((136, 136)), (x, y + 8))
        results[phase] = {char: {'generated_pairwise_pixel_mae': pairwise_mae(generated[char]),
                                 'reference_pairwise_pixel_mae': pairwise_mae(references[char])}
                          for char in targets}
        del pipeline, model, modules
        torch.cuda.empty_cache()
        print(phase + ' ' + json.dumps(results[phase]), flush=True)
    board.save(out / 'fixed_noise_comparison.png')
    record = {'protocol': 'Post-run diagnostic only; same target and seed per target across writers, reference character 人; no checkpoint tuning.',
              'writers': writers, 'targets': targets, 'seeds': [30000, 30001, 30002],
              'results': results,
              'note': 'Pairwise pixel response measures reference sensitivity, not writer-style fidelity. References have different pixels; all generated images only are public.',
              'files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.png'))}}
    (out / 'metrics.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
