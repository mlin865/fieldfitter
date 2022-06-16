# import math
import os
import unittest
from fieldfitter.fitter import Fitter
from opencmiss.utils.zinc.finiteelement import evaluate_field_nodeset_range
from opencmiss.zinc.field import Field


here = os.path.abspath(os.path.dirname(__file__))


def assertAlmostEqualList(testcase, actualList, expectedList, delta):
    assert len(actualList) == len(expectedList)
    for actual, expected in zip(actualList, expectedList):
        testcase.assertAlmostEqual(actual, expected, delta=delta)


class FieldFitterTestCase(unittest.TestCase):

    def test_fit_cube(self):
        """
        Test fitting a cubic Hermite field over a cube.
        """
        zinc_model_file = os.path.join(here, "resources", "cube.exf")
        zinc_data_file = os.path.join(here, "resources", "cube_data.exf")
        fitter = Fitter(zinc_model_file, zinc_data_file)
        fitter.setDiagnosticLevel(2)
        fitter.load()
        self.assertEqual(fitter.getModelCoordinatesField().getName(), "coordinates")
        self.assertEqual(fitter.getDataCoordinatesField().getName(), "data_coordinates")
        self.assertEqual([0.0], fitter.getGradient1Penalty())
        fitter.setGradient1Penalty([0.001])
        self.assertEqual([0.001], fitter.getGradient1Penalty())
        self.assertEqual([0.0], fitter.getGradient2Penalty())
        fitter.setGradient2Penalty([0.001])
        self.assertEqual([0.001], fitter.getGradient2Penalty())

        # fit non-time-varying pressure
        fitter.setFitField("pressure", True)
        self.assertTrue(fitter.fitField("pressure"))
        # fit temperature with 3 times
        fitter.setFitField("temperature", True)
        self.assertTrue(fitter.fitField("temperature"))
        self.assertEqual(fitter.getFieldTimeCount("temperature"), 3)
        times = [0.0, 1.0, 3.0]
        tempTimes = fitter.getFieldTimes("temperature")
        assertAlmostEqualList(self, tempTimes, times, 1.0E-9)

        fieldmodule = fitter.getFieldmodule()
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        datapoints = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        hostLocation = fitter.getDataHostLocationField()

        pressure = fieldmodule.findFieldByName("pressure")
        hostPressure = fieldmodule.createFieldEmbedded(pressure, hostLocation)
        deltaPressure = pressure - hostPressure

        TOL = 1.0E-3
        minPressure, maxPressure = evaluate_field_nodeset_range(pressure, nodes)
        self.assertAlmostEqual(minPressure, 74871.3716089433, delta=TOL)
        self.assertAlmostEqual(maxPressure, 115951.48884538251, delta=TOL)

        minDeltaPressure, maxDeltaPressure = evaluate_field_nodeset_range(deltaPressure, datapoints)
        self.assertAlmostEqual(minDeltaPressure, -656.3450771958596, delta=TOL)
        self.assertAlmostEqual(maxDeltaPressure, 483.561326447787, delta=TOL)

        temperature = fieldmodule.findFieldByName("temperature")
        timeField = fieldmodule.createFieldConstant(0.0)
        timeTemperature = fieldmodule.createFieldTimeLookup(temperature, timeField)
        hostTimeTemperature = fieldmodule.createFieldEmbedded(timeTemperature, hostLocation)
        deltaTimeTemperature = timeTemperature - hostTimeTemperature

        TOL = 1.0E-7
        expectedTemperatures = [
            [3.332641568558043, 118.89672562263947, -0.511420907565892, 0.5522989626074803],
            [-0.5467142199110564, 98.50873077510872, -0.39455720483948653, 0.2338105167652813],
            [-5.573997497222709, 99.1166942045121, -1.0503829692698758, 0.8561196559382651],
        ]

        fieldcache = fieldmodule.createFieldcache()
        for timeIndex in range(len(times)):
            time = times[timeIndex]
            timeField.assignReal(fieldcache, [time])
            minTemperature, maxTemperature = evaluate_field_nodeset_range(timeTemperature, nodes)
            minDeltaTemperature, maxDeltaTemperature = evaluate_field_nodeset_range(deltaTimeTemperature, datapoints)
            self.assertAlmostEqual(minTemperature, expectedTemperatures[timeIndex][0], delta=TOL)
            self.assertAlmostEqual(maxTemperature, expectedTemperatures[timeIndex][1], delta=TOL)
            self.assertAlmostEqual(minDeltaTemperature, expectedTemperatures[timeIndex][2], delta=TOL)
            self.assertAlmostEqual(maxDeltaTemperature, expectedTemperatures[timeIndex][3], delta=TOL)


if __name__ == "__main__":
    unittest.main()
