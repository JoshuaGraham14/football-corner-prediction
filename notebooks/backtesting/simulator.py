import pandas as pd

class Simulator:
    def __init__(self, initial_bankroll):
        """
        Initialises simulator with starting credit
        """
        self.initial_bankroll=initial_bankroll
        self.bankroll=initial_bankroll
        self.total_profit= 0
        self.history = []

    def place_bet(self, match_id, odds, stake, actual_outcome):
        """
        Places a bet of size *stake* and updates the bankroll based on w/l
        """
        if stake>self.bankroll:
            print(f"Insufficient bankroll for bet: {match_id}. Skipping...")
            return

        #Deduct stake as soon as bet is placed...
        self.bankroll -= stake  

        if actual_outcome == 1:  # bet wins...
            profit=(stake*odds)-stake  # profit = return - initial stake
            self.bankroll += stake + profit  # return stake + winnings
        else:  # bet loses...
            profit = -stake  # since entire stake is lost

        self.total_profit+=profit #update profit counter

        self.history.append([match_id, odds, stake, profit, actual_outcome,self.bankroll]) #Store in bet history

    def save_results(self, filename="betting_results.csv"):
        """
        Saves history to csv file
        """
        self.history_df =pd.DataFrame(self.history, columns=['match_id', 'odds','stake', 'profit','outcome', 'bankroll'])
        self.history_df.to_csv(filename,index=False)
        print(f"📊 Results saved to {filename}") 


    def print_trade_summary(self):
        """
        Shows summary of all bets placed
        """
        print("\n📊 Betting Trade Summary:")
        print(f"{'Match ID':<10} {'Odds':<10} {'Stake':<10} {'Profit':<10} {'Outcome':<10} {'Bank':<10}")
        for row in self.history:
            match_id, odds,stake,profit, outcome,bank =  row
            print(f"{match_id:<10} {odds:<10.2f} {stake:<10.2f} {profit:<10.2f} {outcome:<10} {bank:<10.2f}")

    def print_summary(self):
        """
        Prints an overall performance summary:
        """
        num_bets=len(self.history)
        total_staked =sum(row[2] for row in self.history)
        roi = (self.total_profit/total_staked)*100
        win_rate = (sum(1 for row in self.history if row[4]==1)/num_bets)*100

        print("\n--- Overall Summary ---")
        print(f"🏦 Initial Bankroll: £{self.initial_bankroll:.2f}")
        print(f"💰 Final Bankroll: £{self.bankroll:.2f}")
        print(f"📈 Total Profit: £{self.total_profit:.2f}")
        print(f"📊 ROI: {roi:.2f}%")
        print(f"✅ Win rate: {win_rate:.2f}% over {num_bets} bets\n")

        #Output str to return for PDF report:
        output_str = f"""
**--- Overall Summary ---**\n
🏦 **Initial Bankroll**: £{self.initial_bankroll:.2f}\n
💰 **Final Bankroll**: £{self.bankroll:.2f}\n
📈 **Total Profit**: £{self.total_profit:.2f}\n
📊 **ROI**: {roi:.2f}%\n
✅ **Win rate**: {win_rate:.2f}% over {num_bets} bets\n
"""
        
        return output_str