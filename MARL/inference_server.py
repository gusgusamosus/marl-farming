from flask import Flask, request, jsonify
import torch
from collections import defaultdict
from cnn_encoder import CNNEncoder
from policy import ActorCritic
from torchvision import transforms
from PIL import Image
import io
import os
from datetime import datetime

app = Flask(__name__)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load pretrained encoder
print("Loading CNN Encoder...")
encoder = CNNEncoder().to(device)
encoder.eval()

# Load trained RL policy
print("Loading RL Policy...")
policy = ActorCritic(obs_dim=512, action_dim=10).to(device)

# Try to find the latest checkpoint
checkpoint_dir = 'checkpoints'
if os.path.exists(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pt')]
    if checkpoints:
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))[-1]
        policy_path = os.path.join(checkpoint_dir, latest_checkpoint)
        try:
            checkpoint = torch.load(policy_path, map_location=device, weights_only=False)
            policy.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded policy from {policy_path} (Episode {checkpoint['episode']})")
        except Exception as e:
            print(f"✗ Error loading checkpoint: {e}")
            print("Using untrained policy")
    else:
        print(f"✗ No checkpoint files found in {checkpoint_dir}")
        print("Using untrained policy")
else:
    print(f"✗ Checkpoint directory '{checkpoint_dir}' not found")
    print("Using untrained policy")

policy.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Class names
class_names = [
    'Tomato_Bacterial_Spot',
    'Tomato_Early_Blight',
    'Tomato_Healthy',
    'Tomato_Late_Blight',
    'Tomato_Leaf_Mold',
    'Tomato_Mosaic_Virus',
    'Tomato_Septoria_Leaf_Spot',
    'Tomato_Spider_mites Two-spotted_spider_mite',
    'Tomato_Target_Spot',
    'Tomato_Yellow_Leaf_Curl_Virus'
]

# Create directories
os.makedirs('received_images', exist_ok=True)

# Counter for received images
image_counter = 0

# Agent states for MARL (NEW)
agent_states = defaultdict(lambda: {
    'last_action': 0,
    'episode_reward': 0.0,
    'step_count': 0,
    'last_disease': None
})


@app.route("/", methods=["GET"])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'device': str(device),
        'encoder': 'ResNet18 (pretrained)',
        'policy': 'ActorCritic',
        'images_received': image_counter,
        'active_agents': len(agent_states)
    }), 200


@app.route("/upload", methods=["POST"])
def upload():
    """
    Simple classification endpoint (existing functionality)
    """
    global image_counter
    try:
        image_bytes = request.data
        if len(image_bytes) == 0:
            return jsonify({'status': 'error', 'message': 'No image data received'}), 400
        
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        img_path = f'received_images/{timestamp}.jpg'
        img.save(img_path)
        image_counter += 1
        
        # Preprocess
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # CNN Encoding
        with torch.no_grad():
            embedding = encoder(img_tensor).squeeze(0)
        
        # RL Policy Prediction
        with torch.no_grad():
            action, log_prob = policy.act(embedding)
        
        predicted_class = class_names[action]
        
        print(f"[{timestamp}] Image #{image_counter} | Predicted: {predicted_class}")
        
        return jsonify({
            'status': 'success',
            'detected': predicted_class,
            'timestamp': timestamp,
            'image_number': image_counter,
            'confidence': float(torch.exp(log_prob).item()) if log_prob is not None else None
        }), 200
        
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route("/agent_step", methods=["POST"])
def agent_step():
    """
    MARL endpoint: Receive observation from agent, return action (NEW)
    """
    global image_counter
    try:
        # Get agent info
        agent_id = int(request.headers.get('Agent-ID', 0))
        last_action = int(request.headers.get('Last-Action', 0))
        
        # Get image
        image_bytes = request.data
        if len(image_bytes) == 0:
            return jsonify({'action': 0, 'reward': 0.0}), 400
        
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        img_path = f'received_images/agent{agent_id}_{timestamp}.jpg'
        img.save(img_path)
        image_counter += 1
        
        # Preprocess
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # CNN Encoding
        with torch.no_grad():
            embedding = encoder(img_tensor).squeeze(0)
        
        # Get disease classification
        with torch.no_grad():
            disease_action, _ = policy.act(embedding)
        
        predicted_disease = class_names[disease_action]
        
        # Determine swarm action based on disease
        swarm_action = get_swarm_action(predicted_disease, agent_id, last_action)
        
        # Calculate reward
        reward = calculate_reward(agent_id, swarm_action, predicted_disease, last_action)
        
        # Update agent state
        agent_states[agent_id]['last_action'] = swarm_action
        agent_states[agent_id]['episode_reward'] += reward
        agent_states[agent_id]['step_count'] += 1
        agent_states[agent_id]['last_disease'] = predicted_disease
        
        print(f"[Agent {agent_id}] Disease: {predicted_disease} | Action: {swarm_action} | Reward: {reward:.2f}")
        
        return jsonify({
            'action': swarm_action,
            'reward': reward,
            'disease': predicted_disease,
            'step_count': agent_states[agent_id]['step_count']
        }), 200
        
    except Exception as e:
        print(f"Error in agent_step: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'action': 0, 'reward': 0.0}), 500


def get_swarm_action(disease, agent_id, last_action):
    """
    Map disease classification to swarm behavior action
    Actions: 0=Normal, 1=Fast, 2=Slow, 3=Flash, 4=Sleep
    """
    if disease == 'Tomato_Healthy':
        return 2  # Slow capture (conserve resources)
    elif 'Blight' in disease or 'Virus' in disease:
        return 1  # Fast capture (critical monitoring)
    elif 'Spot' in disease or 'Mold' in disease:
        return 3  # Flash on (better visibility)
    else:
        return 0  # Normal mode


def calculate_reward(agent_id, action, disease, last_action):
    """
    Reward function for multi-agent coordination
    """
    reward = 0.0
    
    # Reward disease detection
    if disease != 'Tomato_Healthy':
        reward += 5.0
    else:
        reward += 1.0  # Small reward for healthy monitoring
    
    # Penalty for repeating action
    if action == last_action:
        reward -= 2.0
    
    # Efficiency rewards
    if disease == 'Tomato_Healthy' and action == 2:  # Slow on healthy
        reward += 2.0
    elif disease != 'Tomato_Healthy' and action == 1:  # Fast on disease
        reward += 3.0
    
    return reward


@app.route("/agent_status", methods=["GET"])
def agent_status():
    """
    Get status of all active agents (NEW)
    """
    status = {}
    for agent_id, state in agent_states.items():
        status[f"Agent_{agent_id}"] = {
            'last_action': state['last_action'],
            'episode_reward': state['episode_reward'],
            'step_count': state['step_count'],
            'last_disease': state['last_disease']
        }
    return jsonify(status), 200


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Disease Classification + MARL Server")
    print("="*60)
    print(f"Server running on {device}")
    print(f"Encoder: ResNet18 (pretrained on ImageNet)")
    print(f"Policy: ActorCritic (PPO-trained)")
    print(f"Endpoints:")
    print(f"  - /upload (simple classification)")
    print(f"  - /agent_step (MARL with actions)")
    print(f"  - /agent_status (agent monitoring)")
    print(f"Listening on http://0.0.0.0:8000")
    print("="*60 + "\n")
    app.run(host="0.0.0.0", port=8000, debug=False)
