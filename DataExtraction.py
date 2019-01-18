import pandas as pd
import os
from sqlalchemy import create_engine
import sqlite3
import time
import numpy as np
import matplotlib.pyplot as plt
from rdp import rdp


def angle(directions):
    """Return the angle between vectors
    """
    vec2 = directions[1:]
    vec1 = directions[:-1]

    norm1 = np.sqrt((vec1 ** 2).sum(axis=1))
    norm2 = np.sqrt((vec2 ** 2).sum(axis=1))
    cos = (vec1 * vec2).sum(axis=1) / (norm1 * norm2)
    return np.arccos(cos)


def turning_point_identifier(trajectory,x,y):
    simplified_trajectory = rdp(trajectory, epsilon=2)
    sx, sy = simplified_trajectory.T

    # Visualize trajectory and its simplified version.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, 'r--', label='trajectory')
    ax.plot(sx, sy, 'b-', label='simplified trajectory')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc='best')

    # Define a minimum angle to treat change in direction
    # as significant (valuable turning point).
    min_angle = np.pi / 1.0

    # Compute the direction vectors on the simplified_trajectory.
    directions = np.diff(simplified_trajectory, axis=0)
    theta = angle(directions)

    # Select the index of the points with the greatest theta.
    # Large theta is associated with greatest change in direction.
    idx = np.where(theta > min_angle)[0] + 1

    # Visualize valuable turning points on the simplified trjectory.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(sx, sy, 'gx-', label='simplified trajectory')
    ax.plot(sx[idx], sy[idx], 'ro', markersize=7, label='turning points')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc='best')
    plt.show()


def df2sqlite_v2(dataframe, db_name):
    disk_engine = create_engine('sqlite:///' + db_name + '.db')
    dataframe.to_sql(db_name, disk_engine, if_exists='replace', chunksize=1000)

    """Bundan onceki !!!! Bunu unutma updated_stats V3 icin bunu yapmak daha dogru olabilir. Dont know the difference
    #     dataframe.to_sql(db_name, disk_engine ,if_exists='append')"""


def convert_csvgames_to_sqlite(gameIds):
    for gameId in gameIds:
        file_string = "data/tracking_gameId_" + str(gameId) + ".csv"
        print(file_string)
        df2sqlite_v2(pd.read_csv(file_string), str(gameId))


class DataExtraction(object):

    def __init__(self, plays_database):
        conn = sqlite3.connect(plays_database)
        start_time = time.time()

        self.pass_plays = pd.read_sql_query("SELECT * FROM plays WHERE PassResult IS NOT NULL", conn)
        self.run_plays = pd.read_sql_query("SELECT * FROM plays WHERE PassResult IS NULL", conn)

        conn_players = sqlite3.connect("players.db")
        conn_games = sqlite3.connect("games.db")
        self.games = pd.read_sql_query('SELECT * FROM games', conn_games)
        self.players = pd.read_sql_query('SELECT * FROM players', conn_players)

        self.gameIds = list(self.games.gameId.unique())

        conn_game1 = sqlite3.connect("2017090700.db")

        self.game1 = pd.read_sql_query("SELECT * FROM '2017090700'", conn_game1)
        print(len(self.game1))

    #  convert_csvgames_to_sqlite(gameIds)  # one time only

    def pass_plays_in_a_game(self):
        current_game_id = list(self.game1.gameId)[0]
        # print(current_game_id)
        # select pass plays in the particular game
        pass_plays_in_game1 = self.pass_plays.loc[self.pass_plays.gameId == current_game_id]

        pass_play_Ids_in_game1 = list(pass_plays_in_game1.playId)
        # print(len(pass_play_Ids_in_game1))
        # print(len(pass_plays_in_game1))
        pass_play_tracking_data = self.game1.loc[self.game1['playId'].isin(pass_play_Ids_in_game1)]
        # print(len(pass_play_tracking_data))

        game1_pass_plays_tracking_data = pd.merge(pass_play_tracking_data, pass_plays_in_game1, on=['playId'])
        game1_pass_plays_tracking_data = pd.merge(game1_pass_plays_tracking_data,
                                                  self.players[['nflId', 'PositionAbbr']], on='nflId', how='left')

        print(game1_pass_plays_tracking_data.columns)

        return game1_pass_plays_tracking_data

    def personnel_abbrevation(self, data):
        for i in data.index:
            off_personnel = data.at[i, "personnel.offense"]
            positions = off_personnel.split(',')
            rb_and_te = (positions[:2])
            assert len(rb_and_te) == 2
            number_of_rb = rb_and_te[0].strip().split(' ')[0]
            number_of_te = rb_and_te[1].strip().split(' ')[0]

            data.at[i, "personnel.offense"] = number_of_rb + number_of_te

        return data

    def trajectory_calculator(self, data):
        player_play_data = data.loc[np.logical_and(data['playId'] == 2756, data['jerseyNumber'] == 12)]
        player_xy_data = player_play_data[['x', 'y', 'event']].reset_index(drop=True)

        # start tracking data from the ball snap frame
        ballsnap_frame_index = int(player_xy_data.index[player_xy_data['event'] == 'ball_snap'][0])

        player_xy_data_post_snap = player_xy_data.iloc[ballsnap_frame_index:]

        trajectory = np.column_stack((player_xy_data_post_snap['x'], player_xy_data_post_snap['y']))
        return [trajectory,np.asarray(player_xy_data_post_snap['x']),np.asarray(player_xy_data_post_snap['y'])]


DE = DataExtraction("plays.db")
pass_tracking_data = DE.pass_plays_in_a_game()
pass_tracking_data = DE.personnel_abbrevation(pass_tracking_data)
trajectory,x,y = DE.trajectory_calculator(pass_tracking_data)
turning_point_identifier(trajectory,x,y )