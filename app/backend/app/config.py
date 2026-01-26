import os
from dotenv import load_dotenv

# Load environment variables from a .env file in the current directory.
load_dotenv()

ORS_KEY = os.getenv("ORS_KEY")
# Set your ORS key here or in your environment variables.
