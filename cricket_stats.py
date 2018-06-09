import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import operator
import collections
from pandas.tools.plotting import parallel_coordinates


df = pd.read_csv('D:/kaggle/T20_matches_ball_by_ball_data.csv', parse_dates=["date"], low_memory=False)

def get_batting_stats(df):
    '''
    The methods includes the complete process of calculating
    batting statistics of each batsman available in the
    data set
    :param df: Dataframe of complete ball by ball data from ICC
    :return: batting statistics dataframe
    '''
    # Acculmulate each batsman score in each match
    runs_scored = df.groupby(["Match_Id", "Batting_Team", "Striker"], as_index=False)["Run_Scored"].sum()
    # Count all the balls each batsman faced in a match
    balls_faced = df.groupby(["Match_Id", "Batting_Team", "Striker"], as_index=False)["Run_Scored"].count()
    balls_faced.columns = ["Match_Id", "Batting_Team", "Striker", "Balls_Faced"]
    # Merging the two dataframes to make a complete batting scoreboard
    batting_scoreboard = pd.merge(runs_scored, balls_faced,
                                  on=["Match_Id", "Batting_Team", "Striker"], how="left")

    t20_dismissal = df[["Match_Id", "Batting_Team", "Striker", "Dismissal"]]
    t20_dismissal["concat_key"] = t20_dismissal["Match_Id"].map(str) + ":" + t20_dismissal["Striker"]
    t20_dismissal = t20_dismissal.drop_duplicates(subset=["concat_key"], keep="last")
    t20_dismissal = t20_dismissal.drop(labels="concat_key", axis=1)
    t20_dismissal = t20_dismissal.sort_values(["Match_Id", "Batting_Team"])
    t20_dismissal.Dismissal.fillna("not out", inplace=True)

    batting_scoreboard = pd.merge(batting_scoreboard, t20_dismissal,
                                  on=["Match_Id", "Batting_Team", "Striker"], how="left")

    # Get a unique list of batsman from the scoreboard dataframe
    batsman_statistics = pd.DataFrame({"Batsman": batting_scoreboard.Striker.unique()})

    # Compute "Innings" information for each batsman from the scoreboard dataframe
    Innings = pd.DataFrame(batting_scoreboard.Striker.value_counts())
    Innings.reset_index(inplace=True)
    Innings.columns = ["Batsman", "Innings"]

    # Compute "Not outs" information for each batsman from the scoreboard dataframe
    Not_out = batting_scoreboard.Dismissal == "not out"
    batting_scoreboard["Not_out"] = Not_out.map({True: 1, False: 0})
    Not_out = pd.DataFrame(batting_scoreboard.groupby(["Striker"])["Not_out"].sum())
    Not_out.reset_index(inplace=True)
    Not_out.columns = ["Batsman", "Not_out"]

    # Compute "Balls" information for each batsman from the scoreboard dataframe
    Balls = pd.DataFrame(batting_scoreboard.groupby(["Striker"])["Balls_Faced"].sum())
    Balls.reset_index(inplace=True)
    Balls.columns = ["Batsman", "Balls_Faced"]

    # Compute "Runs" information for each batsman from the scoreboard dataframe
    Run_Scored = pd.DataFrame(batting_scoreboard.groupby(["Striker"])["Run_Scored"].sum())
    Run_Scored.reset_index(inplace=True)
    Run_Scored.columns = ["Batsman", "Run_Scored"]

    # Compute "Highest score" information for each batsman from the scoreboard dataframe
    Highest_Score = pd.DataFrame(batting_scoreboard.groupby(["Striker"])["Run_Scored"].max())
    Highest_Score.reset_index(inplace=True)
    Highest_Score.columns = ["Batsman", "Highest_Score"]

    # Compute "Centuries " information for each batsman from the scoreboard dataframe
    Centuries = pd.DataFrame(
        batting_scoreboard.loc[batting_scoreboard.Run_Scored >= 100,].groupby(["Striker"])["Run_Scored"].count())
    Centuries.reset_index(inplace=True)
    Centuries.columns = ["Batsman", "Centuries"]

    # Compute "Half Centuries " information for each batsman from the scoreboard dataframe
    Half_Centuries = pd.DataFrame(batting_scoreboard.loc[(batting_scoreboard.Run_Scored >= 50) &
                                                         (batting_scoreboard.Run_Scored < 100),].groupby(["Striker"])[
                                      "Run_Scored"].count())
    Half_Centuries.reset_index(inplace=True)
    Half_Centuries.columns = ["Batsman", "Half_Centuries"]

    # Merge all the metric to the batsman statitics dataframe
    batsman_statistics = pd.merge(batsman_statistics, Innings, on=["Batsman"], how="left")
    batsman_statistics = pd.merge(batsman_statistics, Not_out, on=["Batsman"], how="left")
    batsman_statistics = pd.merge(batsman_statistics, Balls, on=["Batsman"], how="left")
    batsman_statistics = pd.merge(batsman_statistics, Run_Scored, on=["Batsman"], how="left")
    batsman_statistics = pd.merge(batsman_statistics, Highest_Score, on=["Batsman"], how="left")

    batsman_statistics = pd.merge(batsman_statistics, Centuries, on=["Batsman"], how="left")
    batsman_statistics.Centuries.fillna(0, inplace=True)
    batsman_statistics.Centuries = batsman_statistics.Centuries.astype("int")

    batsman_statistics = pd.merge(batsman_statistics, Half_Centuries, on=["Batsman"], how="left")
    batsman_statistics.Half_Centuries.fillna(0, inplace=True)
    batsman_statistics.Half_Centuries = batsman_statistics.Half_Centuries.astype("int")

    # Compute "Batting average" for each batsman from the scoreboard dataframe
    batsman_statistics["Batting_Average"] = batsman_statistics.Run_Scored / (
    batsman_statistics.Innings - batsman_statistics.Not_out)
    batsman_statistics.loc[batsman_statistics["Batting_Average"] == np.inf, "Batting_Average"] = 0
    batsman_statistics.loc[batsman_statistics["Batting_Average"].isnull(), "Batting_Average"] = 0

    # Compute "Strike rate for each batsman from the scoreboard dataframe
    batsman_statistics["Strike_Rate"] = (batsman_statistics.Run_Scored * 100) / batsman_statistics.Balls_Faced
    batsman_statistics = batsman_statistics.round({"Batting_Average": 2, "Strike_Rate": 2})
    batsman_statistics = batsman_statistics.sort_values(['Run_Scored'], ascending=False)

    return batsman_statistics


def get_bowling_statistics(df):
    '''

    :param df:
    :return:
    '''
    # Creating a column to determine bowling team for each ball bowled
    df["Bowling_Team"] = pd.DataFrame(
        np.where(df.Batting_Team == df.team, df.team2, df.team))

    # Balls bowled by each bowler
    balls_bowled = pd.DataFrame(df["Bowler"].value_counts())
    balls_bowled.index.name = 'Bowler'
    balls_bowled.columns = ["Total_Ball_Bowled"]

    # Calculate Runs given by each bowler
    df["runs_plus_extras"] = df["Run_Scored"] + df["Extras"]
    runs_given = df.groupby(["Bowler"])["runs_plus_extras"].sum()
    runs_given = pd.DataFrame(runs_given)
    runs_given.reset_index()

    # Wickets taken by each bowler
    df["wickets_taken"] = df["Dismissal"].isnull().map({True: 0, False: 1})
    wickets_taken = pd.DataFrame(df.groupby(["Bowler"])["wickets_taken"].sum())
    wickets_taken.reset_index()

    # Calculating major bowling statistics here including
    # bowling average, economy, strike rate, total wickets etc
    bowling_statistics = pd.merge(balls_bowled, runs_given, how="left", left_index=True, right_index=True)
    bowling_statistics = pd.merge(bowling_statistics, wickets_taken, how="left", left_index=True, right_index=True)
    bowling_statistics["Economy"] = bowling_statistics["runs_plus_extras"] / (
    bowling_statistics["Total_Ball_Bowled"] / 6)
    bowling_statistics["Average"] = bowling_statistics["runs_plus_extras"] / bowling_statistics["wickets_taken"]
    bowling_statistics["Overs"] = bowling_statistics["Total_Ball_Bowled"] / 6
    bowling_statistics = bowling_statistics.round({"Economy": 2, "Average": 2, "Overs": 0})
    bowling_statistics.columns = ["Total_Ball_Bowled", "Total Runs", "Total Wickets", "Economy", "Average", "Overs"]
    bowling_statistics = bowling_statistics[["Overs", "Total Runs", "Total Wickets", "Economy", "Average"]]

    return bowling_statistics

def get_net_run_rate(df):
    # Now let's calculate the Net Run Rate (NRR) for each team as a whole
    # Batting rate per over for each team as a whole
    df["Total_Runs"] = df["Run_Scored"] + df["Extras"]  # Adding runs scored + extras to get total runs col
    df["Bowling_Team"] = pd.DataFrame(
        np.where(df.Batting_Team == df.team, df.team2, df.team))

    runs_scored_by_X_team = pd.DataFrame(
        df.groupby(["Batting_Team"])["Total_Runs"].sum())  # Runs scored by each team in total
    balls_played_by_X_team = pd.DataFrame(
        df.groupby(["Bowling_Team"])["Bowling_Team"].count() / 6)  # Overs played by each team in total
    balls_played_by_X_team.columns = ["Overs_Played"]
    batting_rpo = pd.merge(runs_scored_by_X_team, balls_played_by_X_team, left_index=True,
                           right_index=True)  # RATE PER OVER (RPO) in batting
    batting_rpo = pd.DataFrame(batting_rpo["Total_Runs"] / batting_rpo["Overs_Played"])
    batting_rpo.columns = ["Batting_RPO"]

    # Bowling rate per over of each team as a whole
    runs_scored_by_rest_of_the_teams = pd.DataFrame(
        df.groupby(["Bowling_Team"])["Total_Runs"].sum())  # Runs scored against each team in total
    balls_played_by_rest_of_the_teams = pd.DataFrame(df.groupby(["Bowling_Team"])[
                                                         "Match_Id"].unique().str.len()) * 20  # To calculate bowling rate per over total ball (120) are considered instead of balls played
    balls_played_by_rest_of_the_teams.columns = ['Overs_Played']
    bowling_rpo = pd.merge(runs_scored_by_rest_of_the_teams, balls_played_by_rest_of_the_teams, left_index=True,
                           right_index=True)  # RATE PER OVER (RPO) in bowling
    bowling_rpo = pd.DataFrame(bowling_rpo["Total_Runs"] / (bowling_rpo["Overs_Played"]))
    bowling_rpo.columns = ["Bowling_RPO"]

    # NET RUN RATE CALCULATION: RPO of (REQUIRED TEAM) - RPO of all against teams
    # RPO of  (REQUIRED TEAM) = TOTAL RUNS SCORED AGAISNT ALL TEAMS / TOTAL BALLS PLAYED TO SCORE THOSE RUNS
    # RPO of all against teams = TOTAL RUNS SCORED BY ALL OTHER TEAMS AGAINST REQUIRED TEAM / 20*MATCHES
    net_run_rate = pd.merge(batting_rpo, bowling_rpo, left_index=True, right_index=True)
    net_run_rate["Net_Run_Rate"] = net_run_rate["Batting_RPO"] - net_run_rate["Bowling_RPO"]  # Calc of net run rate
    net_run_rate = net_run_rate.sort_values(["Net_Run_Rate"], ascending=False)

    return net_run_rate


def plot_teams_win_pct(df):
    all_teams = df['team'].unique().tolist()

    winning_pct = {}
    for team in all_teams:
        team_matches = df[(df['team'] == team) | (df['team2'] == team)]
        team_match_ids = team_matches['Match_Id'].unique()
        total_matches = len(team_match_ids)
        df_winnings = df[df['winner'] == team]
        total_wins = len(df_winnings['Match_Id'].unique())
        winning_pct[team] = int(total_wins) / int(total_matches) * 100

    winning_pct_sorted = sorted(winning_pct.items(), key=operator.itemgetter(1))
    winning_pct_sorted = collections.OrderedDict(winning_pct_sorted)

    plt.barh(range(len(winning_pct_sorted)), list(winning_pct_sorted.values()), align='center')
    plt.yticks(range(len(winning_pct_sorted)), list(winning_pct_sorted.keys()))
    plt.xlabel('Percentage Wins (%)')
    plt.ylabel('Teams')
    plt.show()



def plot_top5_batsman(df):
    # Top 5 Batsman of T20 cricket in each team
    df_sub = df[['Striker', 'Run_Scored', 'Batting_Team']]
    df_sub['Run_Scored'] = df_sub['Run_Scored'].astype(int)
    x = df_sub.pivot_table(index='Striker', columns='Batting_Team', aggfunc=sum)

    all_teams = df['team'].unique().tolist()

    top5players = {}
    for team in all_teams:
        y = x['Run_Scored'][team]
        y = y.dropna()
        top5players[team] = dict(y.sort_values(ascending=False)[:5])

    df_plot = pd.DataFrame(top5players).stack().reset_index()
    df_plot.columns = ['Player Name', 'Country', 'Total Score']
    parallel_coordinates(df_plot, class_column='Country')
    plt.show()