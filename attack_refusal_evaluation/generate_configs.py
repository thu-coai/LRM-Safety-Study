import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template_path', type=str, required=True, help='Path to the template YAML file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save generated config files')
    parser.add_argument('--model_checkpoints', type=str, nargs='+', required=True, help='List of model checkpoints')
    args = parser.parse_args()

    with open(args.template_path, 'r') as f:
        template = f.read()

    for ckpt in args.model_checkpoints:
        model_name = ckpt.split('/')[-2]
        target_model_path = ckpt  # Use as-is or modify as needed
        res_save_path = f'results/pair/{model_name}-pair_advbench.jsonl'
        config_content = template.replace('{TARGET_MODEL_PATH}', target_model_path).replace('{RES_SAVE_PATH}', res_save_path)
        config_filename = f'pair_{model_name}.yaml'
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, config_filename), 'w') as out_f:
            out_f.write(config_content)
        print(f'Generated {config_filename}')

if __name__ == '__main__':
    main()