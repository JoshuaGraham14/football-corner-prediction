import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from simulator import Simulator
import os

class Backtester(Simulator):
    def __init__(self, odds_file="totalCorner_odds.csv", model_file=None, bankroll=1000,model_type="classification",fixed_bet_percent=0.02):
        """
        Initialise backtester:
        - odds_file = csv containing historical odds (this file is always used)
        - model_file = csv containing model predictions
        - bankroll = starting credit
        - model_type = classification or "regression"
        """
        super().__init__(bankroll) #init parent Simulator class
        
        #Load historical odds 
        self.odds_data=pd.read_csv(odds_file)
        
        if model_file:
            #Load our model-specific predictions:
            self.model_data = pd.read_csv(model_file)

            #Merge with historcal odds via kaggle_id
            self.data =self.odds_data.merge(self.model_data, on="kaggle_id",how="inner")
            self.data= self.data[[
                "kaggle_id", "odds_1_plus_corner", "actual_result",
                "model_predicted_binary" if model_type=="classification" else "model_predicted_lambda",
            ]].dropna(axis=1, how="all")

        else:
            print("Errorr: No model file provided")  
            return

        self.bankroll_history = []
        self.model_type = model_type
        self.fixed_bet_percent = fixed_bet_percent

    def run(self):
        """
        Runs the backtesting simulation:
        - Goes through each row of the prediction dataset
        - Handles bet place depending on model specified
        """

        for _, row in self.data.iterrows():
            match_id = row['kaggle_id']
            odds =row['odds_1_plus_corner']
            actual_outcome=row['actual_result']  #1 if 1+ corner occurred, 0 if not

            if self.model_type =="classification":
                model_pred = row['model_predicted_binary']  # 1 = Bet, 0 = No Bet
                if model_pred==1: #1 = bet -> so place a bet
                    stake = self.get_fixed_bet_size()
                    self.place_bet(match_id, odds, stake,actual_outcome)

            elif self.model_type == "regression":
                lambda_value= row['model_predicted_lambda']
                poisson_prob= 1-stats.poisson.pmf(0, lambda_value)  # Use poission to calculate P(1+ corners) = 1- P(0 corn)
                historical_prob = row['historical_1_plus_corner_prob']

                if poisson_prob > historical_prob:  #only bet if model suggests better odds...
                    stake = self.kelly_stake(odds, poisson_prob) #calc stake using kelly criterion func
                    self.place_bet(match_id, odds, stake,actual_outcome)

            self.bankroll_history.append(self.bankroll)

        #Simulation over... -> print results and summaries
        self.print_trade_summary()
        self.print_summary()
        # self.save_results("betting_results.csv")
        self.display_results()

    def get_fixed_bet_size(self):
        """
        Returns a fixed % of bankroll as the bet size
        """
        return round(self.bankroll*self.fixed_bet_percent, 2)

    def kelly_stake(self, odds, model_prob):
        """
        Calculates stake using theKelly Criterion to maximise long term expected growth
        Reference: https://en.wikipedia.org/wiki/Kelly_criterion
        """
        edge = model_prob-(1/odds)
        kelly_fraction =edge/(1-edge) if edge > 0 else 0 

        #Our stake is fraction of current bankroll (but also cap at 10% of bankroll)...
        return self.bankroll * min(kelly_fraction, 0.1)

    def display_results(self):
        """
        Plots bankroll growth over time using bankroll_history list
        """
        plt.figure(figsize=(10,5))
        plt.plot(self.bankroll_history,label="Bankroll Over Time", color='blue',linewidth=2)
        plt.axhline(y=self.initial_bankroll,color='gray',linestyle='--',label="Starting Bankroll")
        plt.title(f"Backtesting Bankroll Growth ({self.model_type})")
        plt.xlabel("Bets placed")
        plt.ylabel("Bankroll (£)")
        plt.legend()
        plt.grid(True)
        plt.show()

#Classification model:
backtester = Backtester(model_file="classification_predictions.csv", bankroll=1000, model_type="classification")
backtester.run()

#Regression poisson model:
# backtester_poisson = Backtester(model_file="regression_predictions.csv", bankroll=1000, model_type="regression")
# backtester_poisson.run()