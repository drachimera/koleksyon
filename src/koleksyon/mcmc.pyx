# mcmc simulator
import sys
import random
import pandas as pd
import numpy as np
#pd.set_option('display.max_columns', None)
#from cython.parallel import prange  #TODO: It could run faster still
import koleksyon.lib as ll

#perhaps need to consider this later: https://github.com/EveryTimeIWill18/Cython_Repo
class SimulationArray:
    """
    The SimulationArray is a blazing fast data structure for rolling monte carlo dice based on data from the past.
    rows - commonly a set of 'locations' where events happen, or unique 'buckets'
    columns - commonly a set of blocks, dates or something similar that happen on a re-occuring basis
    """
    def __init__(self, df, rowKey, columnKey, amountKey, n_sims, setup=True):
        """
        rowKey = (str) name of the column in df to define the rows for the simulation, most often something like departments, specialties, locations or buckets
        columnKey = (str) name of the column in df to define the columns for the simulation, most often dates, blocks or some other re-occuring thing such that the event happens in the bucket again
        ammountKey = name of the column that has the amount we are trying to simulate (str)
        n_sims = number of simulations to do (integer)
        setup = should we call setup to prepare the data structures for simulation?  default=Yes
        """
        self.df = df
        self.rowKey = str(rowKey)
        self.columnKey = str(columnKey)
        self.amountKey = str(amountKey)
        self.n_sims = int(n_sims)
        if setup == True:
            self.setup_simulation_array()

    def setup_simulation_array(self):
        print("Setting up Simulation Array...")
        self.cast_cols()
        self.allocate_memory()
        self.preprocess_df_sim_array()
        self.prep_dice()

    def cast_cols(self):
        """
        returns a simple dictionary index for the strings in the original dataset... key->number where the number is the index in the numpy arrays
        contained in this object.
        """   
        self.df[self.amountKey] = self.df[self.amountKey].astype(float)
        self.df[[self.rowKey, self.columnKey]] = self.df[[self.rowKey, self.columnKey]].astype(str)  #rows and columns need to be strings, amounts need to be floats
        rws = list(self.df[self.rowKey].unique())
        self.rows = {}
        for i in range(0,len(rws)):
            self.rows[rws[i]] = i  #simple map that gives back an index to the array for a given row string (or a list via rows.keys())
        clm = list(self.df[self.columnKey].unique())
        self.cols = {}
        for i in range(0,len(clm)):
            self.cols[clm[i]] = i #simple map that gives back an index to the array for a given row string (or a list via cols.keys())
        return self.rows, self.cols

    def allocate_memory(self):
        ## Allocate memory
        #numpy arrays for rapid computation
        rws = self.rows.keys()
        clm = self.cols.keys()
        self.events = np.zeros((len(rws),len(clm)),np.int32)   #np array to contain the total number of events, assume 32 byte int is enough
        self.totals = np.zeros((len(rws),len(clm)),np.float64) #np array to contain the 'actuals'/total value of the events, as a double
        self.simulations = np.zeros((self.n_sims, len(rws),len(clm)),np.float64) #np array to contain the simulation results
        self.dice = {}  #multidimentional dictionary for holding the amounts for events
        for row in rws:
            self.dice[row] = {}
            for col in clm:
                self.dice[row][col] = []  #make everything a list for now

    def preprocess_df_sim_array(self):
        print("Preprocessing Data Frame Into Simulation Array...")
        #go through the dataframe one line at a time, updating our data structures as we go
        print(self.df)
        for i in range(len(self.df)):
            if(i%10000 == 0):
                print(i)
            rowdf = self.df.iloc[i]
            #print("RowDF: ________________________")
            #print(rowdf)
            r = rowdf[self.rowKey]           #get string representation of the index of the data found in the column for 'rows' in the original dataframe
            rowidx = self.rows[r]            #get the array index for the string
            c = rowdf[self.columnKey]        #get string representation of the index for the column from original dataframe
            colidx = self.cols[c]
            #update totals in the arrays...
            self.events[rowidx][colidx] = self.events[rowidx][colidx] + 1
            self.totals[rowidx][colidx] = self.totals[rowidx][colidx] + rowdf[self.amountKey]
            #update the dice
            l = self.dice[r][c]
            l.append(rowdf[self.amountKey])
            #if(i>5):
            #    break

    def prep_dice(self):
        #go through the dice and covert all of those lists to numpy arrays
        print("Preping Dice...")
        rws = self.rows.keys()
        clm = self.cols.keys()
        for row in rws:
            for col in clm:
                l = self.dice[row][col]
                self.dice[row][col] = np.array(l)
        return self.dice

    def simulate(self):
        """
        Do the simulation computations...
        """
        print("Beginning Simulations...")  #TODO: break this out into another call
        #optimizing further:
        # https://stackoverflow.com/questions/58916556/cython-prange-with-function-reading-from-numpy-array-memoryview-freezes-why
        sim_count = 0
        for row, ridx in self.rows.iteritems():
            for col, cidx in self.cols.iteritems():
                events = self.events[ridx][cidx]
                for sim in range(0,self.n_sims):
                    die = self.dice[row][col]
                    rolls = np.random.choice(die, events)
                    self.simulations[sim][ridx][cidx] = np.sum(rolls)
                    sim_count = sim_count + 1
                    if(sim_count%10000 == 0):
                        print(sim_count) 
        return self.simulations

    def get_actual(self, rowStr, colStr):
        """
        Mostly a convience method to do the crosswalk so people don't need to figure out the totals array indexing
        Given that:
           - I have a rowStr coresponding to a specific row in the simulation e.g. a specific location
           - I have a colStr coresponding to a specific column in the simulation e.g. a specific date
        Return:
           - the total of the actuals in the log for the given combination
        """
        rowidx = self.rows[rowStr]  #get the array index for the string
        colidx = self.cols[colStr]  # ... dito ...
        return self.totals[rowidx][colidx]
    def get_simulations(self, rowStr, colStr):
        """
        Mostly a convience method to do the crosswalk so people don't need to figure out what the simulation array indexing is under the covers
        Given that:
           - I have a rowStr coresponding to a specific row in the simulation e.g. a specific location
           - I have a colStr coresponding to a specific column in the simulation e.g. a specific date
        Return:
           - a numpy array of all of the simulations that have been done e.g. [x, y, z, ...]
        """
        rowidx = self.rows[rowStr]  #get the array index for the string
        colidx = self.cols[colStr]  # ... dito ...
        return self.simulations[:, rowidx, colidx]
    def calculate_errors(self, rowStr, colStr):
        """
        After the simulation is done, this method returns the absolute error of each simulation point, relative to the actual value as an array of differences
        rowStr - key for the row you want to access, e.g. the department, specialty, bucket, ect.
        colStr - key for the column you want to access, e.g. date, block, ect.
        """
        return abs(self.get_actual(rowStr, colStr) - self.get_simulations(rowStr, colStr))
    def calculate_errors_on_row(self, rowStr):
        errors_on_row = np.array([])
        for col in self.cols:
            error_bucket = self.calculate_errors(rowStr, col)
            if len(errors_on_row) < 1:
                errors_on_row = error_bucket
            else:
                errors_on_row = np.concatenate((errors_on_row, error_bucket), axis=None)
        return errors_on_row
#TODO: make this into an easy to use exploration tool with density plots!  It doesn't really belong in the simulator
#or perhaps it does, if we combine summary stats for the entire simulation as a dataframe!
#    def calculate_summary_stats(self, npErrors):
#        """
#        Convience method to return back summary statistics for a numpy array of errors
#        """
#        erStr = "errors"
#        dfe = pd.DataFrame(data=npErrors, index=[erStr])
#        report = ll.dist_report(dfe, erStr)
#        return report        

    


#events == things that happened, each event has a numerical value that represents what happened... for example
#blocks == a given time window that events happen in
#locations == the places or concepts where events happened, unique identifier

def power_analysis(n_sims, df_agg_gl, rolled_list, lvl ):
    output = pd.DataFrame()
    for i_sim in range(n_sims):
        #print('i_sim',i_sim)
        for d in range(len(df_agg_gl)):
            act_agg_ch = df_agg_gl.iloc[d,3]
            enc_cnt = df_agg_gl.iloc[d,2]
            date = df_agg_gl.iloc[d,0]

            rolled_ch_total=0
            for N in range(enc_cnt):
                rolled_ch = random.choice(rolled_list)
                rolled_ch_total = rolled_ch_total+rolled_ch
            error = abs(act_agg_ch-rolled_ch_total)
            new_row = {'level':lvl, 'SIMULATION':i_sim, 'SERVICE_DATE':date, 'act_vl':act_agg_ch, 'rolled_vl':rolled_ch_total, 'EROOR':error}
            output = output.append(new_row, ignore_index=True)
    return output
    #output.to_csv(lvl+'_100sims'+'_output_parellel.csv')

#def prep_data():
#    print("Loading Data:")
#    ffile = "/data/finance_dda_models_bucket_v1.0.x_R&D_2019_POWER_DETAIL_DEPARTMENT.csv"
#
#    dfr_19 = pd.read_csv(ffile, nrows=100000)
#
#    print("Preprocessing Data:")
#    dfna = dfr_19.fillna('NA')
#
#    print(dfna)
#
#    df = dfna[['GL2_GL3','DEPARTMENT_NAME','SERVICE_DATE','PAT_ENC_CSN_ID_CHARGE','TOT_CHARGE_TX_AMOUNT']].rename(columns={'GL2_GL3':'GL','PAT_ENC_CSN_ID_CHARGE':'ENC','TOT_CHARGE_TX_AMOUNT': 'Charge'})
#    df['level']=df['GL']+'-'+df['DEPARTMENT_NAME']
#    level_list=df.level.unique()
#    df_agg=df.groupby(['SERVICE_DATE','level']).agg({'ENC':'count', 'Charge': 'sum'}).reset_index().rename(columns={'ENC':'ENC_COUNT','Charge': 'Total_charge'})
#
#    #print(df_agg)
#    return level_list, df, df_agg


#def simulate(level_list, df, df_agg, n_sims=10):
#    for lvl in level_list:
#        print('level',lvl)
#        output = pd.DataFrame()
#        #TODO: this is a table scan in a loop!
#        df_agg_gl =df_agg[df_agg['level']==lvl]
#        df_gl=df[df['level']==lvl]
#        rolled_list = df_gl["Charge"].tolist()
#        result = power_analysis(n_sims, df_agg_gl, rolled_list, lvl)
#        print(result)
#    #output.to_csv('100sims_output.csv')


#def main(n_sims):
#    level_list, df, df_agg = prep_data()
#    #simulate(level_list, df, df_agg, n_sims)
#    df = df.drop('GL', 1)
#    df = df.drop('DEPARTMENT_NAME', 1)
#    df = df.drop('ENC', 1)  
#    SimulationArray(df, 'level', 'SERVICE_DATE', 'Charge', n_sims)

#if __name__ == '__main__':
#    main(int(sys.argv[1]))
