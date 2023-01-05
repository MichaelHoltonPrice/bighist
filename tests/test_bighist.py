import unittest
from bighist.bighist import *
import math


class TestBigHist(unittest.TestCase):

    def test_loadSeshatDataset(self):
        # Ensure an error is thrown for an unsupported version
        with self.assertRaises(ValueError):
            _ = loadSeshatDataset(version='Bad Version')

        # Error check the Equinox dataset
        with self.assertRaises(TypeError):
            _ = loadSeshatDataset(version='Equinox', flavor=None)

        with self.assertRaises(ValueError):
            _ = loadSeshatDataset(version='Equinox', flavor='Bad Worksheet')
        
        for worksheet in getEquinoxWorksheets():
            df = loadSeshatDataset(version='Equinox', flavor=worksheet)
            self.assertTrue(df.shape[0] > 0)

        # Error check the PNAS2017 dataset
        with self.assertRaises(TypeError):
            _ = loadSeshatDataset(version='PNAS2017', flavor=None)

        with self.assertRaises(ValueError):
            _ = loadSeshatDataset(version='Equinox', flavor='Bad Flavor')
        
        for flavor in ['Imputations', 'PCs']:
            df = loadSeshatDataset(version='PNAS2017', flavor=flavor)
            self.assertTrue(df.shape[0] > 0)

    def test_StratifiedTimeSeries(self):
        ts = StratifiedTimeSeries.loadSeshatPNAS2017Data()
        self.assertEqual(ts.df.shape, (8280, 13))
        self.assertEqual(len(ts.features), 9)
        self.assertEqual(ts.featureMatrix.shape, (8280, 9))
        self.assertEqual(ts.pcMatrix.shape, (8280, 9))
        self.assertEqual(ts.P.shape, (8280, 9))
        self.assertEqual(len(ts.D), 9)
        self.assertEqual(ts.Q.shape, (9, 9))

        # Iterate over columns to check the scaling
        for cc_num in range(9):
            CC_scaled = ts.featureMatrix[:,cc_num]
            self.assertAlmostEqual(np.mean(CC_scaled), 0, places=8)
            self.assertAlmostEqual(np.std(CC_scaled), 1, places=8)
        
        # Check that we can add a flow analysis. This also tests doFlowAnalysis
        # TODO: test if interpTimes is not input
        interpTimes = np.arange(-9600,1901,100)
        ts.addFlowAnalysis(interpTimes=interpTimes)
        self.assertEqual(ts.movArrayOut.shape, (414, 9, 2))
        self.assertEqual(ts.velArrayOut.shape, (414, 9, 3))
        self.assertEqual(ts.flowInfo.shape, (414, 2))
        self.assertEqual(ts.movArrayOutInterp.shape, (852, 9, 2))
        self.assertEqual(ts.flowInfoInterp.shape, (852, 2))
        save_file = 'PC1PC2_test_plot.png'
        if os.path.exists(save_file):
            os.remove(save_file)
        ts.plotPC1PC2Movements()
        plt.savefig(save_file)
        self.assertTrue(os.path.exists(save_file))
        #os.remove(save_file)

    def test_getRegionDict(self):
        # Ensure an error is thrown for an unsupported version
        with self.assertRaises(ValueError):
            _ = getRegionDict(version='Bad Version')

        # Check the regionDict for Equinox
        regionDict = getRegionDict('Equinox')
        self.assertEqual(len(regionDict), 10)
        self.assertEqual(len(regionDict['Africa']), 3)
        self.assertEqual(len(regionDict['Europe']), 4)
        self.assertEqual(len(regionDict['Central Eurasia']), 3)
        self.assertEqual(len(regionDict['Southwest Asia']), 5)
        self.assertEqual(len(regionDict['South Asia']), 4)
        self.assertEqual(len(regionDict['Southeast Asia']), 3)
        self.assertEqual(len(regionDict['East Asia']), 3)
        self.assertEqual(len(regionDict['North America']), 4)
        self.assertEqual(len(regionDict['South America']), 3)
        self.assertEqual(len(regionDict['Oceania-Australia']), 3)

        # Check the regionDict for PNAS2017 
        regionDict = getRegionDict('PNAS2017')
        self.assertEqual(len(regionDict), 10)
        for key in regionDict.keys():
            self.assertEqual(len(regionDict[key]), 3)

    def test_getNGAs(self):
        # Ensure an error is thrown for an unsupported version
        with self.assertRaises(ValueError):
            _ = getNGAs(version='Bad Version')

        # Check Equinox
        NGAs = getNGAs(version='Equinox')
        self.assertEqual(len(NGAs), 35)

        # Check PNAS2017 
        NGAs = getNGAs(version='PNAS2017')
        self.assertEqual(len(NGAs), 30)

    def test_tailoredSvd(self):
        # Check that the variance variance explained by PC1 is .772 for 
        # the Seshat PNAS 2017 dataset. tailoredSvd is called inside the
        # init method for StratifiedTimeSeries, so this amounts to an
        # indirect functional nest of the method.

        ts = StratifiedTimeSeries.loadSeshatPNAS2017Data()
        Dsqr = [v**2 for v in ts.D]
        PC1_var = Dsqr[0] / np.sum(Dsqr)
        self.assertAlmostEqual(PC1_var, .772, places=3)

if __name__ == '__main__':
    unittest.main()