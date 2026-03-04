import json
import os
import webdataset as wds
from tqdm import tqdm

# Input and output paths
jsonl_file = '/Users/sengpei.liew/work/Megatron-Energon/tmp/qwen_vl/data/caption.jsonl'
output_dir = '/lustre/users/spliew/megatron_bridge_vlm/dataset/Qwen-Caption'

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Function to read JSONL
def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

# Convert to WDS
print(f"Converting {jsonl_file} to WDS format...")
with wds.ShardWriter(os.path.join(output_dir, 'caption-%d.tar'), maxcount=10000) as shard_writer:
    for entry in tqdm(read_jsonl(jsonl_file)):
        image_path = entry['image']
        
        # Check if the path is absolute or relative. 
        # In the provided sample it was absolute: /lustre/share/downloaded/dataset/coyo-700m/...
        # If it's relative, you might need a base directory.
        
        try:
            with open(image_path, "rb") as img_file:
                image_data = img_file.read()
            
            # Modify conversations as requested
            modified_conversations = []
            for msg in entry.get('conversations', []):
                new_msg = msg.copy()
                if msg['from'] == 'human':
                    new_msg['value'] = ""
                elif msg['from'] == 'gpt':
                    # Prepend "<image> " to the gpt value
                    new_msg['value'] = f"<image> {msg['value']}"
                modified_conversations.append(new_msg)

            sample = {
                "__key__": str(entry['id']),
                "jpg": image_data,
                "json": json.dumps(modified_conversations).encode("utf-8"),
            }
            shard_writer.write(sample)
        except FileNotFoundError:
            print(f"Warning: Image not found at {image_path}. Skipping.")
        except Exception as e:
            print(f"Error processing entry {entry.get('id', 'unknown')}: {e}")

print(f"Dataset successfully converted to WDS at {output_dir}")
