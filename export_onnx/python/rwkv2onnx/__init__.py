import argparse
import os
import sys
import tempfile
import subprocess

from rich import print

from rwkv2onnx_python import fr_to_onnx
from .convert_to_fr import convert_to_fr


def convert(input_path, output_path, dtype, chatrwkv_repo_path):
    cr_convert_script_path = os.path.join(chatrwkv_repo_path, 'v2', 'convert_model.py')
    if not os.path.exists(cr_convert_script_path):
        raise RuntimeError(f'ChatRWKV path "{chatrwkv_repo_path}" does not contain v2/convert_model.py')
    with tempfile.TemporaryDirectory() as tmpdirname:
        print('Converting... (Powered by [link=https://github.com/daquexian/faster-rwkv]Faster RWKV[/link])')
        print('')
        # must end with .pth
        cr_converted_file = os.path.join(tmpdirname, 'cr_converted.pth')
        fr_file = os.path.join(tmpdirname, 'fr')
        subprocess.check_call([sys.executable, cr_convert_script_path, '--in', input_path, '--out', cr_converted_file, '--strategy', f'cpu {dtype}', '--quiet'])
        convert_to_fr(cr_converted_file, fr_file)
        fr_to_onnx(fr_file, output_path, dtype)
        print(f'Finish! Three files are generated:')
        print(f'   "{output_path}" and "{output_path}.bin" are the ONNX model and weights respectively.')
        print(f'   "{output_path}.config" is the metadata file (only needed if you are using Faster RWKV ONNXRuntime backend).')


def main():
    parser = argparse.ArgumentParser(description='Convert ChatRWKV model to ONNX')
    parser.add_argument('input_path', help='Path to input ChatRWKV model file')
    parser.add_argument('output_path', help='Path to output ONNX file')
    parser.add_argument('chatrwkv_path', help='Path to ChatRWKV')
    parser.add_argument('--dtype', help='Data type of the model (fp32 or fp16, default fp16)', default='fp16')
    args = parser.parse_args()
    convert(args.input_path, args.output_path, args.dtype, args.chatrwkv_path)

