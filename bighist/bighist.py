import pandas as pd
import pkg_resources
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import copy

def tailoredSvd(data):
    """A util method to call numpy to do an SVD after removing the mean
    """
    data -= np.mean(data, axis=0)
    P, D, Q = np.linalg.svd(data, full_matrices=False)
    PC_matrix = np.matmul(data, Q.T)
    return P, D, Q, PC_matrix

def loadPNAS2017Data():
    """Load the data from the 2017 PNAS article.

    Keyword arguments:
    scale -- Whether to scale the CC array (default False, do not scale)
    """
    CC_df = loadSeshatDataset(version='PNAS2017', flavor='Imputations')
    CC_names = ['PolPop','PolTerr', 'CapPop',
                'levels', 'government','infrastr',
                'writing', 'texts', 'money']
    CC_matrix_unscaled = CC_df.loc[:, CC_names].values
    CC_matrix_scaled = StandardScaler().fit_transform(CC_matrix_unscaled)
    P, D, Q, PC_matrix = tailoredSvd(CC_matrix_scaled)
    
    return CC_df, CC_names, CC_matrix_unscaled, CC_matrix_scaled,\
        P, D, Q, PC_matrix
 
def loadSeshatDataset(version, flavor=None):
    """Load a seshat dataset

    Keyword arguments:
    version -- The version of the dataset ('PNAS2017' or 'Equinox')
    flavor -- For 'Equinox', the worksheet.
    """

    # Do error checking
    if not version in ['PNAS2017', 'Equinox']:
        raise ValueError('Unrecognized version = ' + version)
    
    data_dir = pkg_resources.resource_filename('bighist', 'data/')
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

def doFlowAnalysis(CC_df, PC_matrix, stratifyBy='NGA', interpTimes=None):
    """Build a movement and velocity arrays for the first two columns of
    PC_matrix. In principle, PC_matrix could be any array based on CC_df, but
    it is probably the principle components. CC_df and PC_matrix must have the
    same number of rows. The unit of analysis is specified by stratifyBy, which
    is probably NGA, but for flexibility can be any column in CC_df. In what is
    obviously a recurring theme, CC_df could be just about any dataframe, but
    probably contains complexity characteristics.
    """

    # The unique units of analysis. We'll call these NGAs, but in principle
    # they could be anythinng
    NGAs = list(set(CC_df[stratifyBy].values))

    # Note: the original Seshat dataset (PNAS2017) used 20 imputations for each
    # entry, but the Equinox dataset only has 1 imputation for each entry. This
    # method allows an arbitrary number of imputations (including none), though
    # with no imputations (or only one) some inefficient averages are done.

    # The inputs for this data creation are the complexity characteristic
    # dataframe, CC_df [8280 x 13; this is for the PNAS2017 dataset], and the
    # matrix of principal component projections, PC_matrix [8280 x 9; this is
    # for the PNAS2017 dataset]. For the PNAS2017 dataset, each row is
    # an imputed observation for 8280 / 20 = 414 unique polity configurations.
    # CC_df provides key information for each observation, such as NGA and
    # Time.
    #
    #  Four arrays are created: movArrayOut, velArrayIn, movArrayIn, and
    # velArrayIn. For the PNAS2017 dataset, all four arrays have the dimensions
    # 414 x 9 x 2. mov stands for movements and vel for velocity. 414 is the
    # numbers of observations, 8 is the number of PCs, and the final axis has
    # two elements: (a) the PC value and (b) the change in the PC value going
    # to the next point in the NGA's time sequence (or, for vel, the change
    # divided by the time difference). The "Out" arrays give the movement (or
    # velocity) away from a point and the "In" arrays give the movement (or
    # velocity) towards a point. The difference is set to NA for the last point
    # in each "Out" sequence and the first point in each "In" sequence. In
    # addition, NGA name and time are stored in the dataframe flowInfo (the
    # needed "supporting" info for each  observation).
    num_cc = PC_matrix.shape[1]

    # Generate the "Out" datasets
    # Initialize the movement array "Out" 
    movArrayOut = np.empty(shape=(0,num_cc,2))
    # Initialize the velocity array "Out" [location, movement / duration,
    #                                      duration]
    velArrayOut = np.empty(shape=(0,num_cc,3))
    # Initialize the info dataframe
    flowInfo = pd.DataFrame(columns=[stratifyBy,'Time']) 

    # Iterate over NGAs to populate movArrayOut, velArrayOut, and flowInfo
    for nga in NGAs:
        indNga = CC_df[stratifyBy] == nga # boolean vector for slicing by NGA
        # Vector of unique times:
        times = sorted(np.unique(CC_df.loc[indNga,'Time']))
        for i_t,t in enumerate(times):
            # boolean vector for slicing also by time:
            ind = indNga & (CC_df['Time']==t)
            newInfoRow = pd.DataFrame(data={'NGA': [nga], 'Time': [t]})
            #flowInfo = flowInfo.append(newInfoRow,ignore_index=True)
            flowInfo = pd.concat([flowInfo, newInfoRow],ignore_index=True)
            newArrayEntryMov = np.empty(shape=(1,num_cc,2))
            newArrayEntryVel = np.empty(shape=(1,num_cc,3))
            for p in range(movArrayOut.shape[1]):
                # Average across imputations:
                newArrayEntryMov[0,p,0] = np.mean(PC_matrix[ind,p])
                # Average across imputations:
                newArrayEntryVel[0,p,0] = np.mean(PC_matrix[ind,p])
                if i_t < len(times) - 1:
                    nextTime = times[i_t + 1]
                    # boolean vector for slicing also by time:
                    nextInd = indNga & (CC_df['Time']==nextTime)
                    nextVal = np.mean(PC_matrix[nextInd,p])
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
    flowInfoInterp = pd.DataFrame(columns=[stratifyBy,'Time']) # Initialize the info dataframe
    if interpTimes is None:
        return movArrayOut, velArrayOut, flowInfo
    
    for nga in NGAs:
        # boolean vector for slicing by NGA:
        indNga = CC_df["NGA"] == nga
        # Vector of unique times:
        times = sorted(np.unique(CC_df.loc[indNga,'Time']))
        for i_t,t in enumerate(interpTimes):
            # Is the time in the NGAs range?
            if t >= min(times) and t <= max(times):
                newInfoRow = pd.DataFrame(data={'NGA': [nga], 'Time': [t]})
                #flowInfoInterp = flowInfoInterp.append(newInfoRow,ignore_index=True)
                flowInfoInterp = pd.concat([flowInfoInterp, newInfoRow],
                                           ignore_index=True)
                newArrayEntry = np.empty(shape=(1,num_cc,2))
                for p in range(movArrayOutInterp.shape[1]):
                    # Interpolate using flowArray
                    indFlow = flowInfo['NGA'] == nga
                    tForInterp = np.array(flowInfo['Time'][indFlow],dtype='float64')
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