*****************************************************
# Chess Neural Network vs. Stockfish
*****************************************************

This project uses a **Python-based Chess Neural Network** that plays against **Stockfish**. It's built for Linux, but if you want to run it on **Windows**, you’ll need to make a few changes.

*****************************************************
## Setup Instructions
*****************************************************

### Linux Setup

1. **Install Stockfish**  
   Download and install the **Linux version** of Stockfish from [here](https://stockfishchess.org/download/).

2. **Clone the Repository**  
   Run these commands in your terminal:
   git clone https://github.com/your-repository/chess-neural-network.git
   cd chess-neural-network

3. **Install Dependencies**

   $pip install -r requirements.txt 

    

   **&&**

   run the game
   *****************************************************
   **If you want to run our current model**
   
   python3 Predict.ipynb                         

   *****************************************************

   
   *****************************************************
   **Train the Model**

   python3 Train.ipynb

      *****************************************************                         

   **But make sure to download and add a pgn game from [lichess](https://lichess.org/)
   and add the downloaded pgn games to chess-engine/engines/torch/models**
   
This project is an enhanced version of NikolAI Skripko's tutorial, now incorporating reinforcement learning. The AI improves by receiving rewards based on its performance with each move, allowing it to learn and adapt over time. Additionally, a low-poly 3D render of the project is in the works, which will visually showcase the AI’s gameplay and decision-making process.

A Windows release is also coming soon, expanding the project to be fully compatible across platforms.
