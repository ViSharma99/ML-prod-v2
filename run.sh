#!/bin/bash


# Activate the virtual environment
conda create -n asos-exercise python==3.7
conda activate asos-exercise

# Install the necessary packages
pip install -r requirements.txt

echo "Do you wish to train the model before starting the API? (yes/no)"
read answer

if [ "$answer" == "yes" ]; then
    # Navigate to the project directory just in case the script isn't run from there

    # Activate the virtual environment (optional, depending on your setup)
    # source venv/bin/activate

    # Run the classifier script to train the model
    python src/classifier.py

    # Deactivate the virtual environment (optional)
    # deactivate
fi



# Start the api server with the --reload flag
cd api || exit
uvicorn main:app --reload