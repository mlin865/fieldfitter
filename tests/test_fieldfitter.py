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
        fitter.setFitField("temperature", True)
        self.assertEqual([0.0], fitter.getGradient1Penalty())
        fitter.setGradient1Penalty([0.001])
        self.assertEqual([0.001], fitter.getGradient1Penalty())
        self.assertEqual([0.0], fitter.getGradient2Penalty())
        fitter.setGradient2Penalty([0.001])
        self.assertEqual([0.001], fitter.getGradient2Penalty())
        self.assertTrue(fitter.fitField("temperature"))

        fieldmodule = fitter.getFieldmodule()
        temperature = fieldmodule.findFieldByName("temperature")
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        datapoints = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        hostLocation = fitter.getDataHostLocationField()
        hostTemperature = fieldmodule.createFieldEmbedded(temperature, hostLocation)
        deltaTemperature = temperature - hostTemperature

        TOL = 1.0E-7
        minTemperature, maxTemperature = evaluate_field_nodeset_range(temperature, nodes)
        self.assertAlmostEqual(minTemperature, 3.332641568558043, delta=TOL)
        self.assertAlmostEqual(maxTemperature, 118.89672562263947, delta=TOL)

        minDeltaTemperature, maxDeltaTemperature = evaluate_field_nodeset_range(deltaTemperature, datapoints)
        self.assertAlmostEqual(minDeltaTemperature, -0.511420907565892, delta=TOL)
        self.assertAlmostEqual(maxDeltaTemperature, 0.5522989626074803, delta=TOL)


if __name__ == "__main__":
    unittest.main()
