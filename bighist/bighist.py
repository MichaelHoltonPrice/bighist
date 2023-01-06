import pandas as pd
import pkg_resources
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import copy
from matplotlib import pyplot as plt



def loadSeshatDataset(version, flavor=None):
    """Load a seshat dataset

    Keyword arguments:
    version -- The version of the dataset ('PNAS2017' or 'Equinox')
    flavor -- For 'Equinox', the worksheet.
    """

    # Do error checking
    if not version in ['PNAS2017', 'Equinox']:
        raise ValueError('Unrecognized version = ' + version)
    
    data_dir = pkg_resources.resource_filename(__name__, 'data/')
    if version == 'Equinox':
        if flavor is None:
            raise TypeError('flavor (worksheet) must be specified for the\
                Equinox dataset')

        worksheets = getEquinoxWorksheets()
        if not flavor in worksheets:
            raise ValueError('Unrecognized Equinox worksheet (flavor) = '\
                 + flavor)

        file_path = os.path.join(data_dir, 'Equinox_on_GitHub_June9_2022.xlsx')
        return pd.read_excel(file_path, sheet_name=flavor)

    if version == 'PNAS2017':
        if flavor is None:
            raise TypeError('flavor (Imputations or PCs) must be specified for\
                the PNAS2017 dataset')

        if not flavor in ['Imputations', 'PCs']:
            raise ValueError('For PNAS2017, the flavor must be Imputations or\
                PCs')

        if flavor == 'Imputations':
            file_name = 'data1.csv'
        elif flavor == 'PCs':
            file_name = 'data1.csv'
        else:
            raise Exception('This should not happen')
        file_path = os.path.join(data_dir, file_name)
        return pd.read_csv(file_path)

class StratifiedTimeSeries:
    """A class to work with a time series containg sub-series (the strata).
    There may be more than one observation per time for each sub-series (e.g.,
    multiple imputations to handle uncertainty and/or missing data).

    Attributes:
        df (dataframe): A Pandas dataframe containing the core data
        features (list): A list of columns in df that are primary features
        timeColumn (str): The column in df that is the time variable
        subSeriesColumn (str): The column in df that marks sub-series
        featureMatrix (numpy array): A matrix of features, normalied to have
            a mean of 0 and standard deviation of 1.
        subseriesToKeep (list): An optional list of subseries to keep (i.e.,
            only keep rows of df that are in subseriesToKeep)
    """
    def __init__(self, df, features, timeColumn,
                 subseriesColumn, subseriesToKeep=None):
        # If necessary, filter using subseriesToKeep
        if subseriesToKeep is not None:
            df = df[df[subseriesColumn].isin(subseriesToKeep)]
        # Store the attributes
        self.df = df
        self.features = features
        self.timeColumn = timeColumn
        self.subseriesColumn = subseriesColumn

        # Create the normalized feature matrix
        self.featureMatrix = self.df.loc[:, self.features].values
        self.featureMatrix =\
            StandardScaler().fit_transform(self.featureMatrix)
        
        # Do a singular value decomposition to get principle components
        P, D, Q, pcMatrix = tailoredSvd(self.featureMatrix)
        self.P = P
        self.D = D
        self.Q = Q
        self.pcMatrix = pcMatrix

    def addFlowAnalysis(self, interpTimes=None):
        if interpTimes is None:
            movArrayOut, velArrayOut, flowInfo =\
                doFlowAnalysis(self.df,
                               self.pcMatrix,
                               self.subseriesColumn,
                               self.timeColumn)
        else:
            movArrayOut, velArrayOut, flowInfo,\
                movArrayOutInterp, flowInfoInterp =\
                    doFlowAnalysis(self.df,
                                   self.pcMatrix,
                                   self.subseriesColumn,
                                   self.timeColumn,
                                   interpTimes=interpTimes)
            self.movArrayOutInterp = movArrayOutInterp
            self.flowInfoInterp = flowInfoInterp
        self.movArrayOut = movArrayOut
        self.velArrayOut = velArrayOut
        self.flowInfo = flowInfo

    def plotPC1PC2Movements(self,
                            lineWidth=.01,
                            headWidth=.15,
                            headLength=.1,
                            arrowAlpha=1,
                            subseriesColors=None):
        """Create a movement plot for PC1 and PC2 using movArrayOut. Most plot
        elements, including axis labels, should be set outside this method. The
        exceptions are the optional inputs, lineWidth (width of the lines),
        headWidth (width of the arrows), headLength (length of the arrows), and
        arrowAlpha (transparency of the arrows). subseriesColors is an optional
        input that contains information on how to colors subseries. It is a
        dictionary that maps subseries IDs onto rgb colors. arrowAlpha is not
        applied to subseriesColors.
        """
        for i in range(0, self.movArrayOut.shape[0]):
            if not np.isnan(self.movArrayOut[i,0,1]):
                nga = self.flowInfo[self.subseriesColumn][i]
                if subseriesColors is not None:
                    rgb = subseriesColors[nga]
                else:
                    rgb = (0,0,1,arrowAlpha)
                plt.arrow(self.movArrayOut[i,0,0],
                          self.movArrayOut[i,1,0],
                          self.movArrayOut[i,0,1],
                          self.movArrayOut[i,1,1],
                          width=lineWidth,
                          head_width=headWidth,
                          head_length=headLength,
                          color=rgb)
#                # Next, plot interpolated points (if necessary)
#                # Doing this all very explicitly to make the code clearer
#                dt = self.velArrayOut[i,0,2]
#                if dt > 100:
#                  for n in range(0,int(dt / 100) - 1):
#                    pc1 = movArrayOut[i,0,0] + velArrayOut[i,0,1]*(float(n+1))*100.
#                    pc2 = movArrayOut[i,1,0] + velArrayOut[i,1,1]*(float(n+1))*100.
#                    plt.scatter(pc1,pc2, s=5,  color=rgb)

    # TODO: consider moving Seshat methods into a standalone class
    # @staticmethod
    def buildNewOldWorldSubseriesColors(newWorldCol, oldWorldCol, NGAs):
        """Build a subseriesColors dictionary for plotPC1PC2Movements where
        NGAs in the New World have the color newWorldCol and NGAs in the Old
        World have the color oldWorldCol.
        """

        regionDict = getRegionDict('Equinox')
        newWorldRegions = ['North America',
                           'South America']
 
        oldWorldRegions = ['Africa',
                           'Europe',
                           'Central Eurasia',
                           'Southwest Asia',
                           'South Asia',
                           'Southeast Asia',
                           'East Asia',
                           'Oceania-Australia']
        # TODO: consider putting the following code in a standalone method
        oldWorldNGAs = list()
        newWorldNGAs = list()
        regionDict = getRegionDict('Equinox')
        for region in newWorldRegions:
            for nga in regionDict[region]:
                newWorldNGAs.append(nga)

        for region in oldWorldRegions:
            for nga in regionDict[region]:
                oldWorldNGAs.append(nga)
 
        subseriesColors = dict()
        for nga in NGAs:
            if nga in newWorldNGAs:
                subseriesColors[nga] = newWorldCol
            elif nga in oldWorldNGAs:
                subseriesColors[nga] = oldWorldCol
            else:
                raise ValueError('Unrecoganized nga = ' + str(nga))
        return subseriesColors
 
    # @staticmethod
    def loadSeshatPNAS2017Data(subseriesToKeep=None):
        """Load the data from the 2017 PNAS article. This returns a
        StratifiedTimeSeries object.
        """
        CC_df = loadSeshatDataset(version='PNAS2017', flavor='Imputations')
        CC_names = ['PolPop','PolTerr', 'CapPop',
                    'levels', 'government','infrastr',
                    'writing', 'texts', 'money']
        
        return StratifiedTimeSeries(CC_df, CC_names, 'Time', 'NGA',
                                    subseriesToKeep)

    # @staticmethod
    def loadSeshatEquinoxData(subseriesToKeep=None):
        """Load the Seshat Equinox dataset. This returns a StratifiedTimeSeries
        object.
        """
        CC_df = loadSeshatDataset(version='Equinox', flavor='ImpSCDat')
        CC_names = ['Pop', 'Terr', 'Cap',
                    'Hier', 'Gov', 'Infra',
                    'Info', 'Money']
        
        return StratifiedTimeSeries(CC_df, CC_names, 'Time', 'NGA',
                                    subseriesToKeep)

   # @staticmethod
    def loadSeshatPNAS2017Data():
        """Load the data from the 2017 PNAS article. This returns a
        StratifiedTimeSeries object.
        """
        CC_df = loadSeshatDataset(version='PNAS2017', flavor='Imputations')
        CC_names = ['PolPop','PolTerr', 'CapPop',
                    'levels', 'government','infrastr',
                    'writing', 'texts', 'money']
        
        return StratifiedTimeSeries(CC_df, CC_names, 'Time', 'NGA')


    # @staticmethod
    def createGridForPC1PC2(dGrid, flowArray):
        """Create a grid for PC1 and PC2 based on the grid spacing dGrid
        """
     
        # Remove endpoints
        ind = [True if not np.isnan(flowArray[i,0,1]) else False\
            for i in range(flowArray.shape[0])]
        fa = flowArray[ind,:,:]
        points2D = fa[:,range(0,2),0]
    
        u0Min = np.floor(np.min(points2D[:,0] - dGrid) / dGrid) * dGrid # PC1 min
        u0Max = np.ceil(np.max(points2D[:,0] + dGrid) / dGrid) * dGrid # PC1 max
        v0Min = np.floor(np.min(points2D[:,1] - dGrid) / dGrid) * dGrid # PC1 min
        v0Max = np.ceil(np.max(points2D[:,1] + dGrid) / dGrid) * dGrid # PC1 max
        u0Vect = np.arange(u0Min,u0Max,dGrid)
        v0Vect = np.arange(v0Min,v0Max,dGrid)
        return u0Vect, v0Vect

def getEquinoxWorksheets():
    """A utility function for getting the worksheets in the Equinox Excel file
    """
    return ['Metadata',
            'Equinox2020_CanonDat',
            'CavIronHSData',
            'HistYield+',
            'TSDat123',
            'AggrSCWarAgriRelig',
            'ImpSCDat',
            'SPC_MilTech',
            'Polities',
            'Variables',
            'NGAs',
            'Scale_MI',
            'Class_MI']

def tailoredSvd(data):
    """A util method to call numpy to do an SVD after removing the mean
    """
    data -= np.mean(data, axis=0)
    P, D, Q = np.linalg.svd(data, full_matrices=False)
    PC_matrix = np.matmul(data, Q.T)
    return P, D, Q, PC_matrix


def getRegionDict(version):
    """Get a dictionary where the key is the region (e.g., 'Africa') and the
    entries are lists of NGAs

    Keyword arguments:
    version -- Which version is this for? e.g., PNAS2017 or Equinox
    """

    # There are any number of ways this method could be implemented. I have
    # chosen to simply create the full regionDict directly for each case.
    regionDict = dict()

    if version == 'Equinox':
        regionDict["Africa"] =\
            ["Ghanaian Coast",
             "Niger Inland Delta",
             "Upper Egypt"]
        regionDict["Europe"] =\
            ["Iceland",
             "Crete", # new
             "Latium",
             "Paris Basin"]
        regionDict["Central Eurasia"] =\
            ["Lena River Valley",
             "Orkhon Valley",
             "Sogdiana"]
        regionDict["Southwest Asia"] =\
            ["Galilee", # new
             "Konya Plain",
             "Southern Mesopotamia", # new
             "Susiana",
             "Yemeni Coastal Plain"]
        regionDict["South Asia"] =\
            ["Garo Hills",
             "Deccan",
             "Kachi Plain",
             "Middle Ganga"] # new
        regionDict["Southeast Asia"] =\
            ["Cambodian Basin",
             "Central Java",
             "Kapuasi Basin"]
        regionDict["East Asia"] =\
            ["Kansai",
             "Southern China Hills",
             "Middle Yellow River Valley"]
        regionDict["North America"] =\
            ["Cahokia",
             "Basin of Mexico", # new
             "Finger Lakes",
             "Valley of Oaxaca"]
        regionDict["South America"] =\
            ["Cuzco",
             "Lowland Andes",
             "North Colombia"]
        regionDict["Oceania-Australia"] =\
            ["Big Island Hawaii",
             "Chuuk Islands",
             "Oro PNG"]
    elif version == 'PNAS2017':
        regionDict["Africa"] =\
            ["Ghanaian Coast",
             "Niger Inland Delta",
             "Upper Egypt"]
        regionDict["Europe"] =\
            ["Iceland",
             "Latium",
             "Paris Basin"]
        regionDict["Central Eurasia"] =\
            ["Lena River Valley",
             "Orkhon Valley",
             "Sogdiana"]
        regionDict["Southwest Asia"] =\
            ["Konya Plain",
             "Susiana",
             "Yemeni Coastal Plain"]
        regionDict["South Asia"] =\
            ["Garo Hills",
             "Deccan",
             "Kachi Plain"]
        regionDict["Southeast Asia"] =\
            ["Cambodian Basin",
             "Central Java",
             "Kapuasi Basin"]
        regionDict["East Asia"] =\
            ["Kansai",
             "Southern China Hills",
             "Middle Yellow River Valley"]
        regionDict["North America"] =\
            ["Cahokia",
             "Finger Lakes",
             "Valley of Oaxaca"]
        regionDict["South America"] =\
            ["Cuzco",
             "Lowland Andes",
             "North Colombia"]
        regionDict["Oceania-Australia"] =\
            ["Big Island Hawaii",
             "Chuuk Islands",
             "Oro PNG"]
    else:
        raise ValueError('Unrecognized version = ' + version)
    return regionDict
 
def getNGAs(version):
    """Get a list of all NGAs.

    Keyword arguments:
    version -- Which version is this for? e.g., PNAS2017 or Equinox
    """
    # The core functionality is in getRegionDict
    regionDict = getRegionDict(version)
    NGAs = list()
    for key in regionDict.keys():
        for nga in regionDict[key]:
            NGAs.append(nga)
    
    NGAs.sort()
    return NGAs

def doFlowAnalysis(df, pcMatrix, subseriesColumn,
                   timeColumn, interpTimes=None):
    """Build a movement and velocity arrays for the input pcMatrix, whicch
    could be original features or principle components. Likely doFlowAnalysis
    is being called by StratifiedTimeSeries.addFlowAnalysis, and the
    documentation for StratifiedTimeSeries contextualizes df, pcMatrix, and
    subseriesColumn.
    """

    # The unique subseries labels. We'll call these NGAs since that is what
    # they would probably be for a seshat analysis.
    NGAs = list(set(df[subseriesColumn].values))

    # Note: the original Seshat dataset (PNAS2017) used 20 imputations for each
    # entry, but the Equinox dataset only has 1 imputation for each entry. This
    # method allows an arbitrary number of imputations (including none), though
    # with no imputations (or only one) some inefficient averages are done.

    # For the Seshat PNAS 2017 analysis, df is the complexity characteristics
    # and pcMatrix is the matrix of PC loadings for the observed data; if so,
    # df has dimensions [8280 x 13] and pcMatrix has dimensions [8280 x 9]. For
    # the PNAS2017 dataset, each row is an imputed observation for
    # 8280 / 20 = 414 unique polity configurations.
    #
    # Four arrays are created: movArrayOut, velArrayIn, movArrayIn, and
    # velArrayIn. For the PNAS2017 dataset, all four arrays have the dimensions
    # 414 x 9 x 2. mov stands for movements and vel for velocity. 414 is the
    # numbers of observations, 9 is the number of PCs / features, and the
    # final axis has two elements: (a) the PC value and (b) the change in the
    # PC value going to the next point in the NGA's time sequence (or, for
    # vel, the change divided by the time difference). The "Out" arrays give
    # the movement (or velocity) away from a point and the "In" arrays give the
    # movement (or velocity) towards a point. The difference is set to NA for
    # the last point in each "Out" sequence and the first point in each "In"
    # sequence. In addition, NGA name and time are stored in the dataframe
    # flowInfo (the needed "supporting" info for each  observation).
    #
    # Optionally, flow objects can be created at a set of input times that are
    # different from the times in df. These times are specified by interpTimes,
    # which is None by default (interp stands for interpolate).
    num_cc = pcMatrix.shape[1]

    # Generate the "Out" datasets
    # Initialize the movement array "Out" 
    movArrayOut = np.empty(shape=(0,num_cc,2))
    # Initialize the velocity array "Out" [location, movement / duration,
    #                                      duration]
    velArrayOut = np.empty(shape=(0,num_cc,3))
    # Initialize the info dataframe
    flowInfo = pd.DataFrame(columns=[subseriesColumn, timeColumn]) 

    # Iterate over NGAs to populate movArrayOut, velArrayOut, and flowInfo
    for nga in NGAs:
        indNga = df[subseriesColumn] == nga # boolean vector for slicing by NGA
        # Vector of unique times:
        times = sorted(np.unique(df.loc[indNga, timeColumn]))
        for i_t,t in enumerate(times):
            # boolean vector for slicing also by time:
            ind = indNga & (df[timeColumn]==t)
            newInfoRow = pd.DataFrame(data={subseriesColumn: [nga],
                                            timeColumn: [t]})
            flowInfo = pd.concat([flowInfo, newInfoRow],ignore_index=True)
            newArrayEntryMov = np.empty(shape=(1,num_cc,2))
            newArrayEntryVel = np.empty(shape=(1,num_cc,3))
            for p in range(movArrayOut.shape[1]):
                # Average across imputations:
                newArrayEntryMov[0,p,0] = np.mean(pcMatrix[ind,p])
                # Average across imputations:
                newArrayEntryVel[0,p,0] = np.mean(pcMatrix[ind,p])
                if i_t < len(times) - 1:
                    nextTime = times[i_t + 1]
                    # boolean vector for slicing also by time:
                    nextInd = indNga & (df[timeColumn]==nextTime)
                    nextVal = np.mean(pcMatrix[nextInd,p])
                    newArrayEntryMov[0,p,1] = nextVal - newArrayEntryMov[0,p,0]
                    newArrayEntryVel[0,p,1] =\
                        newArrayEntryMov[0,p,1]/(nextTime-t)
                    newArrayEntryVel[0,p,2] = (nextTime-t)
                else:
                    newArrayEntryMov[0,p,1] = np.nan
                    newArrayEntryVel[0,p,1] = np.nan
                    newArrayEntryVel[0,p,2] = np.nan
            movArrayOut = np.append(movArrayOut,newArrayEntryMov,axis=0)
            velArrayOut = np.append(velArrayOut,newArrayEntryVel,axis=0)
    
    # Next, create interpolated arrays by iterating over NGAs
    movArrayOutInterp = np.empty(shape=(0,num_cc,2)) # Initialize the flow array 
    # Initialize the info dataframe:
    flowInfoInterp = pd.DataFrame(columns=[subseriesColumn, timeColumn])
    if interpTimes is None:
        return movArrayOut, velArrayOut, flowInfo
    
    for nga in NGAs:
        # boolean vector for slicing by NGA:
        indNga = df[subseriesColumn] == nga
        # Vector of unique times:
        times = sorted(np.unique(df.loc[indNga, timeColumn]))
        for i_t,t in enumerate(interpTimes):
            # Is the time in the NGAs range?
            if t >= min(times) and t <= max(times):
                newInfoRow = pd.DataFrame(data={subseriesColumn: [nga],
                                                timeColumn: [t]})
                #flowInfoInterp = flowInfoInterp.append(newInfoRow,ignore_index=True)
                flowInfoInterp = pd.concat([flowInfoInterp, newInfoRow],
                                           ignore_index=True)
                newArrayEntry = np.empty(shape=(1,num_cc,2))
                for p in range(movArrayOutInterp.shape[1]):
                    # Interpolate using flowArray
                    indFlow = flowInfo[subseriesColumn] == nga
                    tForInterp = np.array(flowInfo[timeColumn][indFlow],
                                          dtype='float64')
                    pcForInterp = movArrayOut[indFlow,p,0]
                    currVal = np.interp(t,tForInterp,pcForInterp)
                    newArrayEntry[0,p,0] = currVal
                    if i_t < len(interpTimes) - 1:
                        nextTime = interpTimes[i_t + 1]
                        nextVal = np.interp(nextTime,tForInterp,pcForInterp)
                        newArrayEntry[0,p,1] = nextVal - currVal
                    else:
                        newArrayEntry[0,p,1] = np.nan
                movArrayOutInterp = np.append(movArrayOutInterp,newArrayEntry,axis=0)

    return movArrayOut, velArrayOut, flowInfo,\
        movArrayOutInterp, flowInfoInterp

