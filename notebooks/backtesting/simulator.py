import pandas as pd
import statistics

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


    def print_trade_log(self, show_output=True):
        """
        Shows log of all bets placed
        """
        if show_output:
            print("\n📊 Betting Trade Summary:")
            print(f"{'Match ID':<10} {'Odds':<10} {'Stake':<10} {'Profit':<10} {'Outcome':<10} {'Bank':<10}")
            for row in self.history:
                match_id, odds,stake,profit, outcome,bank =  row
                print(f"{match_id:<10} {odds:<10.2f} {stake:<10.2f} {profit:<10.2f} {outcome:<10} {bank:<10.2f}")

    def print_summary(self, show_output=True):
        """
        Prints an overall performance summary:
        """
        num_bets=len(self.history)
        total_staked =sum(row[2] for row in self.history)
        roi = (self.total_profit/self.initial_bankroll)*100
        win_rate = (sum(1 for row in self.history if row[4]==1)/num_bets)*100

        # Compute the edge over bookies
        implied_prob = 1/(self.odds_data["odds_1_plus_corner"].mean())*100
        edge = win_rate - implied_prob

        # Sharpe Ratio (shows risk-adjusted return)
        returns = [row[3] / row[2] for row in self.history] #profit divide by stake
        mean_return = statistics.mean(returns)
        std_dev_return = statistics.stdev(returns)
        sharpe_ratio = mean_return / std_dev_return if std_dev_return != 0 else 0

        #Three outputs...

        #1- Output to print to console
        if show_output:
            print("\n--- Overall Summary ---")
            print(f"🏦 Initial Bankroll: £{self.initial_bankroll:.2f}")
            print(f"💰 Final Bankroll: £{self.bankroll:.2f}")
            print(f"💸 Total Staked: £{total_staked:.2f}")
            print(f"📈 Total Profit: £{self.total_profit:.2f}")
            print(f"📊 ROI: {roi:.2f}%")
            print(f"✅ Win rate: {win_rate:.2f}% over {num_bets} bets")
            print(f"📉 Sharpe Ratio: {sharpe_ratio:.2f}\n")
            print(f"🎯 Edge Over Bookies: {edge:.2f}%\n")

        #2 - Output str to return for PDF report (using markdown formatting):
        results_str_list = [
            "**--- Overall Summary ---**",
            f"🏦 **Initial Bankroll**: £{self.initial_bankroll:.2f}",
            f"💰 **Final Bankroll**: £{self.bankroll:.2f}",
            f"💸 **Total Staked**: £{total_staked:.2f}",
            f"📈 **Total Profit**: £{self.total_profit:.2f}",
            f"📊 **ROI**: {roi:.2f}%",
            f"✅ **Win rate**: {win_rate:.2f}% over {num_bets} bets",
            f"📉 **Sharpe Ratio**: {sharpe_ratio:.2f}",
            f"🎯 **Edge Over Bookies**: {edge:.2f}%"
        ]

        #3 - Output as dict for report saving
        results_dict = {
            "initial_bankroll": round(self.initial_bankroll, 2),
            "final_bankroll": round(self.bankroll, 2),
            "total_staked": round(total_staked, 2),
            "total_profit": round(self.total_profit, 2),
            "roi": round(roi, 2),
            "win_rate": round(win_rate, 2),
            "num_bets": round(num_bets, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "edge": round(edge, 2)
        }

        return results_str_list, results_dict