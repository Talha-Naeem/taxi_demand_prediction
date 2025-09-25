from dagshub.upload import Repo
repo = Repo("Talha-Naeem", "Taxi_Demand_Predictor", token="b5373841481ceffc275cd4e0029d2a704116173b") 
repo.upload(local_path="~/Documents/LLM Work/taxi_demand_predict",  
            remote_path="https://dagshub.com/Talha-Naeem/Taxi_Demand_Predictor",  
            commit_message="Taxi Demand Predictor ",)





pipx run dagshub upload Talha-Naeem/Taxi_Demand_Predictor /home/talha-naeem/Documents/LLM Work/taxi_demand_predict https://dagshub.com/Talha-Naeem/Taxi_Demand_Predictor