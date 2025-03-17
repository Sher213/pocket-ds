from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class AnalysisPredictor():
    def __init__(self, dataset):
        self.dataset = f"datasets/{dataset}"
        
        self.problem = None
        self.model = None

        self.messages = []
    
    def prompt(self, prompt):
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content

    def prompt_with_history(self, prompt):
        self.messages.append([
                {
                    "role": "user",
                    "content": prompt
                }
            ])
        return prompt(self.messages)

    def predict_problem(self):
        data = pd.read_csv(self.dataset)

        message = f"""{list(data.columns)}
                    {list(data.iloc[0])}
                    This is a dataset with one row of data. Predict whether it is a:
                    1) Regression Problem
                    2) Classification Problem
                    3) Sentiment Analysis Problem
                    4) Time-Series Problem.

                    Determine and return the corresponding number. Return only the number.
                    """
        
        return self.prompt(message)
    
predictor = AnalysisPredictor(dataset="20250314_144422_insurance.csv")
print(predictor.predict_problem())