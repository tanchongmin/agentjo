# Interactive Text Game
![images](game_view.png)

- Created: 24 Feb 2025

- You can come up with your own scenario, and the game will give you 3 options to choose to advance the story
- If you do not like the options, you can also come up with your own options

- Game keeps track of your location and your emotion for each game state

- Text and image will be generated at each step

- If you do not want the image generation, then choose "Use Default Image"

- Preparation: In your .env file, add in the OPENAI_API_KEY, or the API keys of the LLM you are using

- If you would like to use Pixabay image search instead of generation, add in the PIXABAY_API_KEY in .env. You can get an API key here: <a href = "https://www.pixabay.com/"> https://pixabay.com/ </a>

- There is also an Images and Music folder

- To run: Open the Game.ipynb and run the cell. Enjoy!

You can also customise your game. Example prompts:
## MasterChef
MasterChef Game - You are a chef competing in MasterChef and you seek to impress Gordon Ramsay, one of the judges, so that you may win first place.
![images](masterchef.png)

## Evolution Game
Evolution Game - The game only has two types of options: Choose Ability or Evolve
Player can either choose new abilities for the current organism, or evolve into a new one.
This is meant to be a documentary-style exploration - use realistic organism names and abilities.
Make the exploration full of wonder and possibilities.
The player starts as pieces of nucleotides, and evolves to a human, and then superhuman and beyond. 
Make the emotion neutral, calm or scared only.
Make the location reflect where the organism can live and the name of the organism in the form Location: <location: organism name>
![images](evolution_game.png)
