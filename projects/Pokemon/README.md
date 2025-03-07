Collaborators:
- Dylan Chia
- Dylan Tan
- Nicholas Tan
- Nicholas
- John Tan Chong Min

- This project is inspired by Claude Plays Pokemon (https://www.twitch.tv/claudeplayspokemon).

We feel that the key to learning is in how we consolidate and retrieve memory, and so this is a project to see how far we can push the Agent to learn and play the game on its own.

We're using PokeAgent [https://github.com/DaDevChia/Pokeagent_new](https://github.com/DaDevChia/Pokeagent_new) as the base template to host an API interface for POST and GET requests to the game.

### Sample Python Code for AI Integration
```python
import requests
import base64
from PIL import Image
from io import BytesIO

BASE_URL = "http://localhost:5000/api"

# Initialize the game
requests.post(f"{BASE_URL}/init", json={"rom_path": "PokemonRed.gb"})

# Get the current screen
response = requests.get(f"{BASE_URL}/screen?format=base64")
screen_data = response.json()["image"]
screen_image = Image.open(BytesIO(base64.b64decode(screen_data)))

# Get game state
game_state = requests.get(f"{BASE_URL}/state").json()

# Make a decision based on the state and screen
# ... AI logic here ...

# Execute an action
requests.post(f"{BASE_URL}/button", json={"button": "a"})
```
