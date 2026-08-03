import os
import shutil

'''
    We use the official eval dataset provided by Colosseum Challenge(https://huggingface.co/datasets/colosseum/colosseum-challenge/tree/main).
    However, we find the data format is not that clean. Specifically, we do the following tidyment:
    - Delete the redundant “Copy of stack_cups_6_eval.zip“ file in “stack_cups_6”.
    - For "put_money_in_safe","open_drawer","move_hanger","meat_on_grill","insert_onto_square_peg","hockey","get_ice_from_fridge","empty_dishwasher","basketball_in_hoop", the variation13 folder should be added "variation0" subfolder.

    Or you can download the cleaned data we have tided from https://huggingface.co/datasets/LPY/BridgeVLA_COLOSSUM_EVAL_DATA/tree/main.
'''
def delete_zip_in_stack_cups(root_dir):
    for root, dirs, files in os.walk(root_dir):
        if "stack_cups" in root:
            for file in files:
                if file.endswith('.zip'):
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        print(f"Deleted {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")

def create_variation0_folders(root_dir):
    for root, dirs, files in os.walk(root_dir):
        if root.endswith('_13'):
            variation0_path = os.path.join(root, 'variation0')
            if not os.path.exists(variation0_path):
                try:
                    os.makedirs(variation0_path)
                    print(f"Created {variation0_path}")
                    for item in os.listdir(root):
                        if item != 'variation0':
                            item_path = os.path.join(root, item)
                            new_path = os.path.join(variation0_path, item)
                            shutil.move(item_path, new_path)
                            print(f"Moved {item_path} to {new_path}")
                except Exception as e:
                    print(f"Error processing {root}: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python clean_colosseum_eval_data.py <directory_path>")
        sys.exit(1)
    root_directory = sys.argv[1]
    delete_zip_in_stack_cups(root_directory)
    create_variation0_folders(root_directory)
