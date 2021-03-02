import unittest
import koleksyon.mcmc as mcmc
import pandas as pd
import numpy as np
import datetime

def artist_costs():
    #Artist-Album costs (0.0 represents that you don't buy and albumn, .99 represents you buy just a song)
    MichaelJackson = np.array([0.0,0.99,8.64,8.69,12.33,12.96,38.99,30.12,13.99,17.25])
    LadyGaga = np.array([0.0,0.99,14.28,11.20,11.25,14.98,13.69,9.99,18.95])
    Eminem = np.array([0.0,0.99,15.99,9.33,21.61,22.37,12.80,10.75,11.70])
    JustinBieber = np.array([0.0,0.99,10.70,9.49,14.65,29.18,21.93,15.95,19.90,37.98])
    FreddieMercury = np.array([0.0,0.99,14.74,11.50,18.99,12.49,14.54,10.99,11.89,16.53,11.70,9.71,12.39])
    MileyCyrus = np.array([0.0,0.99,11.18,6.98,9.21,9.95,9.49])
    TaylorSwift = np.array([0.0,0.99,13.98,16.99,13.51,8.97,15.02,7.00,13.97,8.97,6.86])
    LilWayne = np.array([0.0,0.99,11.55,16.00,29.47,13.41,9.68,15.95,11.99,16.63])
    SelenaGomez = np.array([0.0,0.99,12.59,10.91,36.57,16.52])
    Rihanna = np.array([0.0,0.99,13.98,10.25,22.90,6.32,9.19])
    ArtistAlbums = {}
    ArtistAlbums["Michael Jackson"] = MichaelJackson
    ArtistAlbums["Lady Gaga"] = LadyGaga
    ArtistAlbums["Eminem"] = Eminem
    ArtistAlbums["Justin Bieber"] = JustinBieber
    ArtistAlbums["Freddie Mercury"] = FreddieMercury
    ArtistAlbums["Miley Cyrus"] = MileyCyrus
    ArtistAlbums["Taylor Swift"] = TaylorSwift
    ArtistAlbums["Lil Wayne"] = LilWayne
    ArtistAlbums["Selena Gomez"] = SelenaGomez
    ArtistAlbums["Rihanna"] = Rihanna
    return ArtistAlbums

def ppurchase(ArtistAlbums):
    purchase_probability = {}
    for k,v in ArtistAlbums.items():
        #print(k)
        #print(v)
        proba = []
        proba.append(0.70)  #30% purchases, 70% not purchases
        proba.append(0.24)
        r =  0.06 / (len(v) - 2)
        for i in range(0,len(v)-2):
            proba.append(r)
        proba = np.array(proba)
        #print(proba)
        purchase_probability[k] = proba
    return purchase_probability

class TestMCMC(unittest.TestCase):

    def setUp(self):
        self.prepData()
    #@classmethod
    #def setUpClass(cls):
    #    cls.prepData(None)  #note the strange way we need to interact with the class on setup!

    #simple function that preps the data for ALL the tests in this file... runs once... note, random seed set to make reproduceable
    def prepData(self):
        print("Preparing the data so we can run tests...")
        np.random.seed(42)   
        print("Parsing count data and setting up probability arrays")     
        self.ArtistAlbums = artist_costs()
        self.purchase_probability = ppurchase(self.ArtistAlbums)
        self.dfa = pd.read_csv("../data/artist_wiki_page_views-20200101-20201231.csv")  #original count data
        #just filter down for unit tests for three dates, so this runs quick...
        c1 = self.dfa['Date'].str.contains("2020-01-15")
        c2 = self.dfa['Date'].str.contains("2020-02-15")
        c3 = self.dfa['Date'].str.contains("2020-03-15")
        self.dfa = self.dfa[c1 | c2 | c3]
        self.dfa = self.dfa.reset_index()            #now that data is filtered, we need to drop the index for stuff to work
        self.dfa = self.dfa.drop(columns=['index'])
        print("Simulating log data...")
        self.dflog = self.generate_log_data(self.dfa)                                   #simulated purchase data
        return self.dflog, self.dfa

    #helper function for prep data
    #generate a dataframe shaped like a purchase log for a single cell in the page view table
    def generate_amounts(self, artist, date, count):
        amounts = np.random.choice(self.ArtistAlbums[artist], count, p=self.purchase_probability[artist])
        df = pd.DataFrame(data=amounts,  columns=["amount"])
        df['artist'] = artist 
        df['date'] = date
        return df 

    #helper function for prep data
    def generate_log_data(self, df_page_view_counts):
        scale_factor = 1000 #if scale factor == 1, dataset will be true size, 10==one tenth the size, 100=one one hundreth ect.
        #iterate over the page view counts
        dflog = pd.DataFrame(columns=['date', 'artist', 'amount']) #dataframe to place the log
        #print(df_page_view_counts)
        rws, clms = df_page_view_counts.shape
        date_time = datetime.datetime.now()
        for i in range(rws):
            for col in list(df_page_view_counts.columns):
                if 'Date' in col:
                    #print("*************date*********************")
                    #print(df_page_view_counts.at[i,col])
                    date_time = datetime.datetime.strptime(df_page_view_counts.at[i,col], '%Y-%m-%d')
                    #print(date_time)
                    print('.', end='', flush=True)
                else:
                    #print(df.iat[i,j])
                    artist = col
                    visits = int(df_page_view_counts.at[i,col])
                    dfl = self.generate_amounts(artist, date_time, visits // scale_factor)
                    #print(dfl)
                    dflog = dflog.append(dfl, ignore_index=True)
                    #print(len(dflog))
        print("")
        dflog['date'] = pd.to_datetime(dflog['date'])
        return dflog


    def test_simulation_array(self):
        print("Testing Simulation Array Construction...")
        #pd.set_option('display.max_rows', 1000)
        print(self.dflog)
        n_sims = 10
        print("Building Simulator")
        simulator = mcmc.SimulationArray(self.dflog, 'artist', 'date', 'amount', n_sims, setup=False)  #alternate form of constructor for testing/debugging, usually setup=True
        self.assertEqual('artist', simulator.rowKey)
        self.assertEqual('date', simulator.columnKey)
        self.assertEqual('amount', simulator.amountKey)
        self.assertEqual(n_sims, simulator.n_sims)

    def test_cast_cols(self):
        print("Testing Casting of Columns...")
        simulator = mcmc.SimulationArray(self.dflog, 'artist', 'date', 'amount', 10, setup=False)
        drow, dcol = simulator.cast_cols()
        self.assertEqual((476,3), simulator.df.shape)  #ensure we did not do something funny to the original df
        #expected results
        e_row_keys = ['Michael Jackson', 'Lady Gaga', 'Eminem', 'Justin Bieber', 'Freddie Mercury', 'Miley Cyrus', 'Taylor Swift', 'Lil Wayne', 'Selena Gomez', 'Rihanna']
        e_row_vals = [0,1,2,3,4,5,6,7,8,9]
        e_col_keys = ['2020-01-15', '2020-02-15', '2020-03-15']
        e_col_vals = [0,1,2]
        self.assertEqual(e_row_keys, list(drow.keys()))
        self.assertEqual(e_row_vals, list(drow.values()))
        self.assertEqual(e_col_keys, list(dcol.keys()))
        self.assertEqual(e_col_vals, list(dcol.values()))

    def test_allocate_memory(self):
        print("Testing Allocate Memory, where we make the np arrays that hold the result of the simulations")
        simulator = mcmc.SimulationArray(self.dflog, 'artist', 'date', 'amount', 10, setup=False)
        drow, dcol = simulator.cast_cols()  #have to do this, the steps are taken 1 at a time...
        simulator.allocate_memory()
        self.assertEqual((10,3),simulator.events.shape)   #10 by 3 array, all zeros..  this will hold the event counts after the next step
        self.assertTrue(np.all((simulator.events == 0)))  #contains only zeros
        self.assertEqual((10,3),simulator.totals.shape)   #10 by 3 array, all zeros..  this represents the totals for the given artist-date pairs, as a double
        self.assertTrue(np.all((simulator.totals == 0)))  #contains only zeros
        self.assertEqual((10,10,3),simulator.simulations.shape) #10 simulations, each of the same size as the events
        self.assertTrue(np.all((simulator.simulations == 0)))   #contains only zeros
        print(simulator.dice)
        self.assertEqual(10,len(simulator.dice.keys()))             #10 keys, one for each artist
        self.assertEqual(3, len(simulator.dice['Michael Jackson'])) # each artist has 3 dates
        self.assertEqual(0, len(simulator.dice['Michael Jackson']['2020-02-15']))  #for a given date, we currently have an empty list
        #self.dice = {} 

    def test_preprocess_df_sim_array(self):
        print("Testing That the Data in the Numpy Arrays used for the simulation is consistent with what we have in the pandas dataframe")
        simulator = mcmc.SimulationArray(self.dflog, 'artist', 'date', 'amount', 10, setup=False)
        drow, dcol = simulator.cast_cols()  #have to do this, the steps are taken 1 at a time...
        simulator.allocate_memory()         #and this also...
        simulator.preprocess_df_sim_array() #step we are testing...
        #total number of events (including zeros) ... note number of events is bigger if there are more web clicks!
        eevents = np.array( [[17, 18, 15],
                    [10, 13, 14],
                    [12, 32, 18],
                    [15, 45, 12],
                    [16, 33, 28],
                    [10,  9, 10],
                    [11, 26, 14],
                    [ 6, 10,  5],
                    [27, 11, 12],
                    [ 8,  9, 10]] )
        #totals of the orders (including zeros)
        etotals = np.array( [[25.61, 60.95,  0.99],
                    [ 0.99, 12.96, 39.81],
                    [31.93,  6.93,  4.95],
                    [33.14, 32.82, 25.84],
                    [14.67, 83.41, 31.91],
                    [ 4.95,  4.95,  1.98],
                    [ 2.97,  9.97, 49.88],
                    [ 2.97, 15.39,  1.98],
                    [ 8.91, 14.57, 11.9 ],
                    [ 1.98,  2.97, 44.32]] )
        #print(simulator.events)
        #print(type(eevents))
        #print(type(simulator.events))
        self.assertTrue(np.array_equal(eevents, simulator.events))
        #print(simulator.totals)
        self.assertTrue(np.allclose(etotals, simulator.totals))

    def test_prep_dice(self):
        print("Testing that the dice hash is instantiated as a numpy array after the call")
        simulator = mcmc.SimulationArray(self.dflog, 'artist', 'date', 'amount', 10, setup=False)
        drow, dcol = simulator.cast_cols()  #have to do this, the steps are taken 1 at a time...
        simulator.allocate_memory()         #and this also...
        simulator.preprocess_df_sim_array() #this too...
        #print(simulator.dice)
        dice = simulator.prep_dice()        #test that everything is converted to a numpyarray...
        #print(dice)
        for artist in dice.keys():
            for date in dice[artist].keys():
                arr = dice[artist][date]
                self.assertTrue("ndarray", type(arr))

    def test_simulate(self):
        print("Testing we can simulate data with a similar distribution as the original data")
        n_sims = 2 
        simulator = mcmc.SimulationArray(self.dflog, 'artist', 'date', 'amount', 2)  #setup=True, means do all the functions: cast_cols(), allocate_memory(), preprocess_df_sim_array(), prep_dice()  
        simulator.simulate()
        esimulations = np.array([[[ 16.92,  20.97,   0.99],
                        [  0.,    16.92,  14.28],
                        [  3.96,   3.96,   1.98],
                        [ 34.13,  10.89,  45.74],
                        [  1.98, 126.24,  11.88],
                        [  5.94,   2.97,   0.99],
                        [  2.97,   4.95,  63.85],
                        [  2.97,   2.97,   2.97],
                        [  6.93,   1.98,  22.81],
                        [  0.,     4.95,  36.05]],
                        [[ 17.91,  22.95,   1.98],
                        [  0.99,  20.97,  42.84],
                        [ 64.83,   2.97,   1.98],
                        [ 32.15,  30.84,   7.92],
                        [ 14.67,  76.8,   54.91],
                        [  5.94,   5.94,   0.99],
                        [  3.96,  16.97,  67.86],
                        [  2.97,  16.38,   0.99],
                        [  7.92,  14.57,  12.89],
                        [  3.96,   5.94,  36.05]]])
        print(simulator.simulations)
        #we have the seed locked, so the simulation should result in the same array as above
        self.assertTrue(np.allclose(esimulations, simulator.simulations))

        #now that we actually have a simulation result, gonna run a bunch of simple tests on it we can manually check!
        actual_x = simulator.get_actual('Michael Jackson', '2020-02-15')
        self.assertEqual(60.95, actual_x)

        simulations_x = simulator.get_simulations('Michael Jackson', '2020-02-15')
        self.assertTrue( np.allclose( np.array([20.97, 22.95]), simulations_x)) #note, second column first row, from matrix1 and matrix2

        #example of errors calculated for a single day
        errors = simulator.calculate_errors('Michael Jackson', '2020-02-15') #errors for a particular date, look at simulations returned for same above
        self.assertAlmostEqual( abs(60.95 -20.97 ),   errors[0])
        self.assertAlmostEqual( abs(60.95 -22.95 ),   errors[1])

        #example of errors calculated for all days
        errors_on_row = simulator.calculate_errors_on_row("Michael Jackson")
        eerrors_on_row = np.array([ 8.69,  7.7,  39.98, 38.,    0.,    0.99])  #errors on the row regardless of date
        #print("Row Errors: __^^^____")
        #print(errors_on_row)
        self.assertTrue( np.allclose( eerrors_on_row, errors_on_row ) )

        
if __name__ == '__main__':
    unittest.main()