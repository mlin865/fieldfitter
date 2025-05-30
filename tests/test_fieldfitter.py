import logging
import os
import sys
import unittest
from fieldfitter.fitter import Fitter
from cmlibs.utils.zinc.finiteelement import evaluate_field_nodeset_range
from cmlibs.utils.zinc.region import copy_fitting_data
from cmlibs.zinc.context import Context
from cmlibs.zinc.field import Field


here = os.path.abspath(os.path.dirname(__file__))


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def assertAlmostEqualList(testcase, actualList, expectedList, delta):
    assert len(actualList) == len(expectedList)
    for actual, expected in zip(actualList, expectedList):
        testcase.assertAlmostEqual(actual, expected, delta=delta)


class FieldFitterTestCase(unittest.TestCase):

    def test_fit_cube(self):
        """
        Test fitting a cubic Hermite field over a cube.
        """
        # index i chooses one of 2 ways to run the test:
        # 0. From supplied model data files
        # 1. With a user-supplied region into which the user builds model, loads data and performs fit

        zinc_model_file_name = os.path.join(here, "resources", "cube.exf")
        zinc_data_file_name = os.path.join(here, "resources", "cube_data.exf")
        for i in range(2):
            if i == 0:
                # use fitter with model and data files
                fitter = Fitter(zinc_model_file_name, zinc_data_file_name)
                fitter.setDiagnosticLevel(2)
                fitter.load()
                # region = fitter.getRegion()
                fieldmodule = fitter.getFieldmodule()
                # coordinates = fitter.getModelCoordinatesField()
            else:
                # use fitter with user-specified region; caller must build model, load data and set up fit
                context = Context("Scaffoldfitter test")
                region = context.getDefaultRegion()
                fitter = Fitter(region=region)
                fitter.setDiagnosticLevel(2)

                region.readFile(zinc_model_file_name)
                fieldmodule = region.getFieldmodule()
                coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
                self.assertEqual(coordinates.getNumberOfComponents(), 3)
                fitter.setModelCoordinatesField(coordinates)

                dataRegion = region.createChild("raw_data")
                dataRegion.readFile(zinc_data_file_name)
                copy_fitting_data(region, dataRegion)
                data_coordinates = fieldmodule.findFieldByName("data_coordinates").castFiniteElement()
                fitter.setDataCoordinatesField(data_coordinates)
                fitter.updateFitFields()

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
