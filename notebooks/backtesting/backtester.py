import os
import sys
# Set project root
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from simulator import Simulator

class Backtester(Simulator):
    def __init__(self, config, odds_file, model_name="", model_file=None, track_num=1, target_mean=1.32, margin=0.1):
        """
        Initialise backtester:
        - odds_file = csv containing historical odds (this file is always used)
        - model_file = csv containing model predictions
        - bankroll = starting credit
        """
        #Get data from config
        bankroll = int(config["backtesting"]["initial_bankroll"])
        fixed_bet_percent = float(config["backtesting"]["fixed_bet_percent"])

        super().__init__(bankroll) #init parent Simulator class
        
        #Load historical odds 
        self.odds_data=pd.read_csv(odds_file)

        # Scale odds
        self.scale_odds(target_mean, margin)

        if model_file:
            #Load our model-specific predictions:
            self.model_data = pd.read_csv(model_file)

            #Merge with historcal odds via kaggle_id
            self.data =self.odds_data.merge(self.model_data, on="kaggle_id",how="inner")
            self.data= self.data[["kaggle_id", "odds_1_plus_corner", "actual_result", "model_predicted_binary"]].dropna(axis=1, how="all")

        else:
            print("Errorr: No model file provided")  
            return

        self.bankroll_history = []
        self.fixed_bet_percent = fixed_bet_percent
        self.track_num = track_num
        self.model_name = model_name

    def scale_odds(self, target_mean, margin):
        """
        Scales the odds_1_plus_corner to fit within a range centered 
        around target mean.
        Also applies a specified 'bookies' margin...
        """

        # print(f"Mean pre: {self.odds_data['odds_1_plus_corner'].mean()}")
        spread = target_mean * margin
        min_odds = target_mean - spread/2

        min_original=self.odds_data["odds_1_plus_corner"].min()
        max_original=self.odds_data["odds_1_plus_corner"].max()

        self.odds_data["odds_1_plus_corner"] = min_odds + (
            (self.odds_data["odds_1_plus_corner"]-min_original) /
            (max_original-min_original)
        ) * spread
        # print(f"Mean post: {self.odds_data['odds_1_plus_corner'].mean()}")


    def run(self, show_output=True):
        """
        Runs the backtesting simulation:
        - Goes through each row of the prediction dataset
        - Handles bet place depending on model specified
        """

        for _, row in self.data.iterrows():
            match_id = row['kaggle_id']
            odds =row['odds_1_plus_corner']
            actual_outcome=row['actual_result']  #1 if 1+ corner occurred, 0 if not
            model_pred = row['model_predicted_binary']  # 1 = Bet, 0 = No Bet

            if model_pred==1: #1 = bet -> so place a bet
                stake = self.get_fixed_bet_size()
                self.place_bet(match_id, odds, stake,actual_outcome)

            self.bankroll_history.append(self.bankroll)

        #Simulation over... -> print results and summaries
        self.print_trade_log(show_output)
        results_str_list, results_dict = self.print_summary(show_output)
        backtesting_image_path = self.display_results(show_output)

        return backtesting_image_path, results_str_list, results_dict

    def get_fixed_bet_size(self):
        """
        Returns a fixed % of bankroll as the bet size
        """
        return round(self.bankroll*self.fixed_bet_percent, 2)

    def display_results(self, show_output=True):
        """
        Plots bankroll growth over time using bankroll_history list
        """
        plt.figure(figsize=(10,5))
        plt.plot(self.bankroll_history,label="Bankroll Over Time", color='blue',linewidth=2)
        plt.axhline(y=self.initial_bankroll,color='gray',linestyle='--',label="Starting Bankroll")
        plt.title(f"Backtesting Bankroll Growth (Track {self.model_name})")
        plt.xlabel("Bets placed")
        plt.ylabel("Bankroll (£)")
        plt.legend()
        plt.grid(True)

        # Save graph as an image
        backtesting_image_path = f"../reports/images/backtesting/backtesting_track{self.track_num}_{self.model_name}.png"
        plt.savefig(backtesting_image_path)
        
        if show_output:
            plt.show()
        else:
            plt.close()

        return backtesting_image_path
    