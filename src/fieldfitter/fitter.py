"""
Class for fitting fields on geometrically aligned scaffolds.
"""

import json
import sys
from timeit import default_timer as timer

from cmlibs.utils.zinc.field import getGroupList, findOrCreateFieldStoredMeshLocation, getUniqueFieldName, \
    orphanFieldByName
from cmlibs.utils.zinc.finiteelement import getMaximumNodeIdentifier
from cmlibs.utils.zinc.general import ChangeManager
from cmlibs.zinc.context import Context
from cmlibs.zinc.field import Field, FieldFindMeshLocation, FieldFiniteElement
from cmlibs.zinc.node import Node
from cmlibs.zinc.optimisation import Optimisation
from cmlibs.zinc.result import RESULT_OK, RESULT_WARNING_PART_DONE


class Fitter:
    """
    Class for fitting fields on geometrically aligned scaffolds.
    """

    _nodeValueLabels = [Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1,
                        Node.VALUE_LABEL_D_DS2, Node.VALUE_LABEL_D2_DS1DS2,
                        Node.VALUE_LABEL_D_DS3, Node.VALUE_LABEL_D2_DS1DS3,
                        Node.VALUE_LABEL_D2_DS2DS3, Node.VALUE_LABEL_D3_DS1DS2DS3]

    def __init__(self, zincModelFileName: str, zincDataFileName: str):
        """
        :param zincModelFileName: Name of zinc file supplying model to define fields on.
        :param zincDataFileName: Name of zinc file supplying data to fit to.
        """
        self._zincModelFileName = zincModelFileName
        self._zincDataFileName = zincDataFileName
        self._context = Context("Fieldfitter")
        self._zincVersion = self._context.getVersion()[1]
        self._logger = self._context.getLogger()
        self._region = None
        self._rawDataRegion = None
        self._fieldmodule = None
        self._dataCoordinatesField = None
        self._dataCoordinatesFieldName = None
        self._modelCoordinatesField = None
        self._modelCoordinatesFieldName = None
        self._modelFitGroup = None
        self._modelFitGroupName = None
        # fibre field is used to orient strain/curvature penalties. None=global axes
        self._fibreField = None
        self._fibreFieldName = None
        self._diagnosticLevel = 0
        self._fitFields = {}  # map from field name to dict containing per-field parameters, esp. "fit" -> bool
        self._hasFitFields = {}  # map from field name to bool, True if field has been fitted
        self._gradient1Penalty = [0.0]  # up to 3 values in x, y, z
        self._gradient2Penalty = [0.0]  # up to 9 values in xx, xy, xz, yx, yy, yz, zx, zy, zz
        self._dataHostLocationField = None  # stored mesh location field in highest dimension mesh for all data
        self._dataHostCoordinatesField = None  # embedded field giving host coordinates at data location
        self._dataHostDeltaCoordinatesField = None  # self._dataHostCoordinatesField - self._dataCoordinatesField
        self._mesh = []  # [dimension - 1]

    def decodeSettingsJSON(self, s: str):
        """
        Define Fitter from JSON serialisation output by encodeSettingsJSON.
        :param s: String of JSON encoded Fitter settings.
        """
        settings = json.loads(s)
        # field names are read (default to None), fields are found on load
        self._dataCoordinatesFieldName = settings.get("dataCoordinatesField")
        self._modelCoordinatesFieldName = settings.get("modelCoordinatesField")
        self._modelFitGroupName = settings.get("modelFitGroup")
        self._fibreFieldName = settings.get("fibreField")
        self._diagnosticLevel = settings["diagnosticLevel"]
        self._fitFields = settings.get("fitFields")
        self._gradient1Penalty = settings.get("gradient1Penalty")
        self._gradient2Penalty = settings.get("gradient2Penalty")

    def encodeSettingsJSON(self) -> str:
        """
        :return: String JSON encoding of Fitter settings.
        """
        dct = {
            "dataCoordinatesField": self._dataCoordinatesFieldName,
            "modelCoordinatesField": self._modelCoordinatesFieldName,
            "modelFitGroup": self._modelFitGroupName,
            "fibreField": self._fibreFieldName,
            "diagnosticLevel": self._diagnosticLevel,
            "fitFields": self._fitFields,
            "gradient1Penalty": self._gradient1Penalty,
            "gradient2Penalty": self._gradient2Penalty
        }
        return json.dumps(dct, sort_keys=False, indent=4)

    def getContext(self):
        return self._context

    def printLog(self):
        loggerMessageCount = self._logger.getNumberOfMessages()
        if loggerMessageCount > 0:
            for i in range(1, loggerMessageCount + 1):
                print(self._logger.getMessageTypeAtIndex(i), self._logger.getMessageTextAtIndex(i))
            self._logger.removeAllMessages()

    def getDiagnosticLevel(self):
        return self._diagnosticLevel

    def setDiagnosticLevel(self, diagnosticLevel):
        """
        :param diagnosticLevel: 0 = no diagnostic messages. 1 = Information and warning messages.
        2 = Also optimisation reports.
        """
        assert diagnosticLevel >= 0
        self._diagnosticLevel = diagnosticLevel

    def getFieldmodule(self):
        return self._fieldmodule

    def getMesh(self, dimension):
        assert 1 <= dimension <= 3
        return self._mesh[dimension - 1]

    def getMeshHighestDimension(self):
        """
        :return: Highest dimension mesh with elements in it, or None if none.
        """
        for d in range(2, -1, -1):
            mesh = self._mesh[d]
            if mesh.getSize() > 0:
                return mesh
        return None

    def getRegion(self):
        return self._region

    def getZincVersion(self):
        """
        :return: zinc version numbers [major, minor, patch].
        """
        return self._zincVersion

    def load(self):
        """
        Read model and data and define initial projections.
        """
        self._clearFields()
        self._region = self._context.createRegion()
        self._fieldmodule = self._region.getFieldmodule()
        self._rawDataRegion = self._context.createRegion()
        self._loadModel()
        self._loadData()

    def isFitField(self, name: str) -> bool:
        """
        Query whether to fit the field of name.
        :param name: Name of field to query.
        :return: True to fit, False to not fit.
        """
        dct = self._fitFields.get(name)
        assert dct, "FieldFitter.isFitField: Invalid field name"
        return dct["fit"]

    def setFitField(self, name: str, isFit: bool):
        """
        Set whether to fit the field of name. Field is marked for fitting later.
        :param name: Name of field to modify.
        :param isFit: True to fit, False to not fit.
        :return: True on success, False if failed.
        """
        dct = self._fitFields.get(name)
        assert dct, "FieldFitter.setFitField: Invalid field name"
        field = self._fieldmodule.findFieldByName(name)
        if isFit and (field in (self._dataCoordinatesField, self._modelCoordinatesField)):
            dct["fit"] = False
            # print("FieldFitter.  Cannot fit coordinate fields")
            return False
        elif not isFit and self._hasFitFields[name]:
            self._undefineField(field)
        dct["fit"] = isFit
        return True

    def getFitFieldNames(self):
        """
        Get list of field names which could be fit.
        """
        return self._fitFields.keys()

    def isFieldFitted(self, name: str) -> bool:
        """
        Query whether the field of name has been fitted.
        :param name: Name of field to query.
        :return: True to fit, False to not fit.
        """
        isFitted = self._hasFitFields.get(name)
        assert isFitted is not None, "FieldFitter.isFieldFitted: Invalid field name"
        return isFitted

    def fitField(self, name: str) -> bool:
        """
        Fit field of name to data. Marks field for fitting first, which fails if
        field is not valid to fit.
        :param name: Name of field to fit.
        :return: True on success, False if failed to fit.
        """
        if not self.isFitField(name):
            if not self.setFitField(name, True):
                return False
        if self.isFieldFitted(name):
            return True
        field = self._fieldmodule.findFieldByName(name).castFiniteElement()
        return self._defineField(field) and self._fitField(field)

    def undefineField(self, name: str) -> bool:
        """
        Undefine field of name from elements. Reverses fitField().
        """
        if not self.isFieldFitted(name):
            return
        field = self._fieldmodule.findFieldByName(name).castFiniteElement()
        return self._undefineField(field)

    def getGradient1Penalty(self, count: int = None):
        """
        Get list of penalty factors used to scale directional gradients of each
        component of fit field. Up to 3 components possible in 3-D.
        :param count: Optional number of factors to limit or enlarge list to,
        at least 1 if supplied. If enlarging, values are padded with the last
        stored value. If None (default) the number stored is requested.
        :return: list(float).
        """
        if count:
            assert count > 0
            count = min(count, 3)
            storedCount = len(self._gradient1Penalty)
            if count <= storedCount:
                gradient1Penalty = self._gradient1Penalty[:count]
            else:
                lastFactor = self._gradient1Penalty[-1]
                gradient1Penalty = self._gradient1Penalty[:] + [lastFactor]*(count - storedCount)
        else:
            gradient1Penalty = self._gradient1Penalty[:]  # shallow copy
        return gradient1Penalty

    def setGradient1Penalty(self, gradient1Penalty: list):
        """
        Set penalty. Clears all fitted fields if changed.
        :param gradient1Penalty: List of 1-3 float penalty values to apply to first gradient of fit field w.r.t. x, y
        and z. Applies identically to all components of the fit field.
        If fewer than 3 values are supplied in the list, the last value is used in all following directions.
        Pass [0.0] to reset/disable.
        """
        if not isinstance(gradient1Penalty, list):
            print("FieldFitter: setGradient1Penalty requires a list of float", file=sys.stderr)
            return
        gradient1Penalty = gradient1Penalty[:3]  # shallow copy, limiting size
        count = len(gradient1Penalty)
        assert count > 0, "FieldFitter: setGradient1Penalty requires a list of at 1-3 floats"
        for i in range(count):
            assert isinstance(gradient1Penalty[i], float), \
                "FieldFitter: setGradient1Penalty requires a list of float"
            if gradient1Penalty[i] < 0.0:
                gradient1Penalty[i] = 0.0
        if self._gradient1Penalty != gradient1Penalty:
            self._gradient1Penalty = gradient1Penalty
            self._clearFittedFields()

    def getGradient2Penalty(self, count: int = None):
        """
        Get list of penalty factors used to scale 2nd directional gradients of
        each component of fit field. Up to 9 components possible in 3-D.
        :param count: Optional number of factors to limit or enlarge list to,
        at least 1 if supplied. If enlarging, values are padded with the last
        stored value. If None (default) the number stored is requested.
        :return: list(float).
        """
        if count:
            assert count > 0
            count = min(count, 9)
            storedCount = len(self._gradient2Penalty)
            if count <= storedCount:
                gradient2Penalty = self._gradient2Penalty[:count]
            else:
                lastFactor = self._gradient2Penalty[-1]
                gradient2Penalty = self._gradient2Penalty[:] + [lastFactor]*(count - storedCount)
        else:
            gradient2Penalty = self._gradient2Penalty[:]  # shallow copy
        return gradient2Penalty

    def setGradient2Penalty(self, gradient2Penalty: list):
        """
        Set penalty. Clears all fitted fields if changed.
        :param gradient2Penalty: List of 1-9 float penalty values to apply to first gradient of fit field w.r.t.
        xx, xy, xz, yx, yy, yz, zx, zy, zz. Applies identically to all components of the fit field.
        If fewer than 9 values are supplied in the list, the last value is used in all following directions.
        Pass [0.0] to reset/disable.
        """
        if not isinstance(gradient2Penalty, list):
            print("FieldFitter: setGradient2Penalty requires a list of float", file=sys.stderr)
            return
        gradient2Penalty = gradient2Penalty[:9]  # shallow copy, limiting size
        count = len(gradient2Penalty)
        assert count > 0, "FieldFitter: setGradient2Penalty requires a list of at 1-9 floats"
        for i in range(count):
            assert isinstance(gradient2Penalty[i], float), \
                "FieldFitter: setGradient2Penalty requires a list of float"
            if gradient2Penalty[i] < 0.0:
                gradient2Penalty[i] = 0.0
        if self._gradient2Penalty != gradient2Penalty:
            self._gradient2Penalty = gradient2Penalty
            self._clearFittedFields()

    def getDataCoordinatesField(self):
        return self._dataCoordinatesField

    def setDataCoordinatesField(self, dataCoordinatesField: Field, clear=True):
        """
        Set field giving data coordinates.
        :param dataCoordinatesField: Field giving coordinates for data points, matching model coordinates.
        :param clear: True to clear all fitted fields; pass false if calling from initial discovery of field.
        """
        if dataCoordinatesField == self._dataCoordinatesField:
            return
        finiteElementField = dataCoordinatesField.castFiniteElement()
        assert finiteElementField.isValid() and (finiteElementField.getNumberOfComponents() == 3)
        if clear:
            self._clearFittedFields()
        self._dataCoordinatesFieldName = dataCoordinatesField.getName()
        self._dataCoordinatesField = finiteElementField
        self._defineDataEmbedding()

    def setDataCoordinatesFieldByName(self, dataCoordinatesFieldName):
        self.setDataCoordinatesField(self._fieldmodule.findFieldByName(dataCoordinatesFieldName))

    def getDataHostLocationField(self):
        return self._dataHostLocationField

    def getDataHostCoordinatesField(self):
        return self._dataHostCoordinatesField

    def getDataHostDeltaCoordinatesField(self):
        return self._dataHostDeltaCoordinatesField

    def getFibreField(self):
        return self._fibreField

    def setFibreField(self, fibreField: Field, clear=True):
        """
        Set field used to orient 1st and 2nd order smoothing penalties relative to element.
        :param fibreField: Fibre angles field available on elements, or None to use
        global x, y, z axes.
        :param clear: True to clear all fitted fields; pass false if calling from initial discovery of field.
        """
        if fibreField == self._fibreField:
            return
        assert (fibreField is None) or \
            ((fibreField.getValueType() == Field.VALUE_TYPE_REAL) and (fibreField.getNumberOfComponents() <= 3)), \
            "Fieldfitter: Invalid fibre field"
        if clear:
            self._clearFittedFields()
        self._fibreField = fibreField
        self._fibreFieldName = fibreField.getName() if fibreField else None

    def getModelCoordinatesField(self):
        return self._modelCoordinatesField

    def setModelCoordinatesField(self, modelCoordinatesField: Field, clear=True):
        """
        Set field giving model coordinates.
        :param modelCoordinatesField: Field giving coordinates over model, matching data coordinates.
        :param clear: True to clear all fitted fields; pass false if calling from initial discovery of field.
        """
        if modelCoordinatesField == self._modelCoordinatesField:
            return
        finiteElementField = modelCoordinatesField.castFiniteElement()
        mesh = self.getMeshHighestDimension()
        assert finiteElementField.isValid() and (mesh.getDimension() <= finiteElementField.getNumberOfComponents() <= 3)
        if clear:
            self._clearFittedFields()
        self._modelCoordinatesField = finiteElementField
        self._modelCoordinatesFieldName = modelCoordinatesField.getName()
        self._defineDataEmbedding()

    def setModelCoordinatesFieldByName(self, modelCoordinatesFieldName):
        self.setModelCoordinatesField(self._fieldmodule.findFieldByName(modelCoordinatesFieldName))

    def getModelFitGroup(self):
        """
        :return: Zinc FieldGroup or None
        """
        return self._modelFitGroup

    def setModelFitGroup(self, modelFitGroup):
        """
        Set group to define and fit field over. Clears all fitted fields if changed.
        :param modelFitGroup: Zinc FieldGroup to limit fit to, or None to fit to whole mesh.
        """
        if modelFitGroup == self._modelFitGroup:
            return
        assert (modelFitGroup is None) or modelFitGroup.castGroup().isValid()
        self._clearFittedFields()  # must do first as iterates over previous group
        if modelFitGroup is None:
            self._modelFitGroup = None
            self._modelFitGroupName = None
        else:
            self._modelFitGroup = modelFitGroup.castGroup()
            self._modelFitGroupName = modelFitGroup.getName()

    def fitAllFields(self):
        """
        Fit all currently non-fitted fields.
        """
        for fieldName in self._fitFields:
            if self.isFitField(fieldName):
                self.fitField(fieldName)

    def writeFittedFields(self, fittedFieldsFileName):
        """
        Write fitted fields over fitted group of model.
        """
        with ChangeManager(self._fieldmodule):
            sir = self._region.createStreaminformationRegion()
            sir.setRecursionMode(sir.RECURSION_MODE_OFF)
            srf = sir.createStreamresourceFile(fittedFieldsFileName)
            if self._modelFitGroupName:
                sir.setResourceGroupName(srf, self._modelFitGroupName)
            fieldNames = []
            for fieldName in self._fitFields:
                if self.isFitField(fieldName):
                    fieldNames.append(fieldName)
            sir.setResourceFieldNames(srf, fieldNames)
            sir.setResourceDomainTypes(srf, Field.DOMAIN_TYPE_NODES |
                                       Field.DOMAIN_TYPE_MESH1D | Field.DOMAIN_TYPE_MESH2D | Field.DOMAIN_TYPE_MESH3D)
            result = self._region.write(sir)
            assert result == RESULT_OK

    def writeData(self, fileName):
        """
        Write just the data in model region for diagnostic purposes.
        """
        sir = self._region.createStreaminformationRegion()
        sir.setRecursionMode(sir.RECURSION_MODE_OFF)
        sr = sir.createStreamresourceFile(fileName)
        sir.setResourceDomainTypes(sr, Field.DOMAIN_TYPE_DATAPOINTS)
        self._region.write(sir)

    def _clearFields(self):
        self._dataCoordinatesField = None
        self._modelCoordinatesField = None
        self._modelFitGroup = None
        self._fibreField = None
        self._dataHostLocationField = None
        self._dataHostCoordinatesField = None
        self._dataHostDeltaCoordinatesField = None
        self._mesh = []  # [dimension - 1]

    def _clearFittedFields(self):
        """
        Clear the flags indicating any fields have been fitted.
        """
        for name in self._fitFields:
            if self._hasFitFields[name]:
                field = self._fieldmodule.findFieldByName(name).castFiniteElement()
                self._undefineField(field)

    def _loadModel(self):
        result = self._region.readFile(self._zincModelFileName)
        assert result == RESULT_OK, "Failed to load model file" + str(self._zincModelFileName)
        self._mesh = [self._fieldmodule.findMeshByDimension(d + 1) for d in range(3)]
        self._discoverModelCoordinatesField()
        self._discoverModelFitGroup()
        self._discoverFibreField()

    def _loadData(self):
        """
        Load zinc data file into self._rawDataRegion.
        Rename data groups to exactly match model groups where they differ by case and whitespace only.
        Transfer data points (and converted nodes) into self._region.
        """
        result = self._rawDataRegion.readFile(self._zincDataFileName)
        assert result == RESULT_OK, "Failed to load data file " + str(self._zincDataFileName)
        dataFieldmodule = self._rawDataRegion.getFieldmodule()
        with ChangeManager(dataFieldmodule):
            # rename data groups to match model
            # future: match with annotation terms
            modelGroupNames = [group.getName() for group in getGroupList(self._fieldmodule)]
            writeDiagnostics = self.getDiagnosticLevel() > 0
            for dataGroup in getGroupList(dataFieldmodule):
                dataGroupName = dataGroup.getName()
                compareName = dataGroupName.strip().casefold()
                for modelGroupName in modelGroupNames:
                    if modelGroupName == dataGroupName:
                        if writeDiagnostics:
                            print("Load data: Data group '" + dataGroupName + "' found in model")
                        break
                    elif modelGroupName.strip().casefold() == compareName:
                        result = dataGroup.setName(modelGroupName)
                        if result == RESULT_OK:
                            if writeDiagnostics:
                                print("Load data: Data group '" + dataGroupName + "' found in model as '" +
                                      modelGroupName + "'. Renaming to match.")
                        else:
                            print("Error: Load data: Data group '" + dataGroupName + "' found in model as '" +
                                  modelGroupName + "'. Renaming to match FAILED.")
                            if dataFieldmodule.findFieldByName(modelGroupName).isValid():
                                print("    Reason: field of that name already exists.")
                        break
                else:
                    if writeDiagnostics:
                        print("Load data: Data group '" + dataGroupName + "' not found in model")
            # if there are both nodes and datapoints, offset datapoint identifiers to ensure different
            nodes = dataFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
            if nodes.getSize() > 0:
                datapoints = dataFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
                if datapoints.getSize() > 0:
                    maximumDatapointIdentifier = max(0, getMaximumNodeIdentifier(datapoints))
                    maximumNodeIdentifier = max(0, getMaximumNodeIdentifier(nodes))
                    # this assumes identifiers are in low ranges and can be improved if there is a problem:
                    identifierOffset = 100000
                    while (maximumDatapointIdentifier > identifierOffset) or (maximumNodeIdentifier > identifierOffset):
                        assert identifierOffset < 1000000000, "Invalid node and datapoint identifier ranges"
                        identifierOffset *= 10
                    while True:
                        # logic relies on datapoints being in identifier order
                        datapoint = datapoints.createNodeiterator().next()
                        identifier = datapoint.getIdentifier()
                        if identifier >= identifierOffset:
                            break
                        result = datapoint.setIdentifier(identifier + identifierOffset)
                        assert result == RESULT_OK, "Failed to offset datapoint identifier"
                # transfer nodes as datapoints to self._region
                sir = self._rawDataRegion.createStreaminformationRegion()
                srm = sir.createStreamresourceMemory()
                sir.setResourceDomainTypes(srm, Field.DOMAIN_TYPE_NODES)
                self._rawDataRegion.write(sir)
                result, buffer = srm.getBuffer()
                assert result == RESULT_OK, "Failed to write nodes"
                buffer = buffer.replace(bytes("!#nodeset nodes", "utf-8"), bytes("!#nodeset datapoints", "utf-8"))
                sir = self._region.createStreaminformationRegion()
                sir.createStreamresourceMemoryBuffer(buffer)
                result = self._region.read(sir)
                assert result == RESULT_OK, "Failed to load nodes as datapoints"
        # transfer datapoints to self._region
        sir = self._rawDataRegion.createStreaminformationRegion()
        srm = sir.createStreamresourceMemory()
        sir.setResourceDomainTypes(srm, Field.DOMAIN_TYPE_DATAPOINTS)
        self._rawDataRegion.write(sir)
        result, buffer = srm.getBuffer()
        assert result == RESULT_OK, "Failed to write datapoints"
        sir = self._region.createStreaminformationRegion()
        sir.createStreamresourceMemoryBuffer(buffer)
        result = self._region.read(sir)
        assert result == RESULT_OK, "Failed to load datapoints"
        self._discoverDataCoordinatesField()
        self._updateFitFields()

    def _updateFitFields(self):
        """
        Build fitFields list, merging any previous stored settings.
        Reset flag marking whether field is fitted.
        """
        oldFitFields = self._fitFields
        self._fitFields = {}
        fieldIter = self._fieldmodule.createFielditerator()
        field = fieldIter.next()
        while field.isValid():
            if field.isManaged() and (field.getValueType() == Field.VALUE_TYPE_REAL) \
                    and field.castFiniteElement().isValid():
                name = field.getName()
                dct = {"fit": False}
                oldDct = oldFitFields.get(name)
                if oldDct:
                    dct.update(oldDct)
                self._fitFields[name] = dct
                self._hasFitFields[name] = False
            field = fieldIter.next()

    def _discoverDataCoordinatesField(self):
        """
        Choose default dataCoordinates field.
        """
        self._dataCoordinatesField = None
        field = None
        if self._dataCoordinatesFieldName:
            field = self._fieldmodule.findFieldByName(self._dataCoordinatesFieldName)
        if not (field and field.isValid()):
            datapoints = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
            datapoint = datapoints.createNodeiterator().next()
            if datapoint.isValid():
                fieldcache = self._fieldmodule.createFieldcache()
                fieldcache.setNode(datapoint)
                fielditer = self._fieldmodule.createFielditerator()
                field = fielditer.next()
                while field.isValid():
                    if field.isTypeCoordinate() and (field.getNumberOfComponents() == 3) and \
                            (field.castFiniteElement().isValid()):
                        if field.isDefinedAtLocation(fieldcache):
                            break
                    field = fielditer.next()
                else:
                    field = None
        self.setDataCoordinatesField(field, clear=False)

    def _discoverFibreField(self):
        """
        Find field used to orient strain and curvature penalties, if any.
        """
        self._fibreField = None
        fibreField = None
        # guarantee a zero fibres field exists
        zeroFibreFieldName = "zero fibres"
        zeroFibreField = self._fieldmodule.findFieldByName(zeroFibreFieldName)
        if not zeroFibreField.isValid():
            with ChangeManager(self._fieldmodule):
                zeroFibreField = self._fieldmodule.createFieldConstant([0.0, 0.0, 0.0])
                zeroFibreField.setName(zeroFibreFieldName)
                zeroFibreField.setManaged(True)
        if self._fibreFieldName:
            fibreField = self._fieldmodule.findFieldByName(self._fibreFieldName)
        if not (fibreField and fibreField.isValid()):
            fibreField = None  # in future, could be zeroFibreField?
        self.setFibreField(fibreField, clear=False)

    def _discoverModelCoordinatesField(self):
        """
        Choose default modelCoordinates field.
        """
        self._modelCoordinatesField = None
        field = None
        if self._modelCoordinatesFieldName:
            field = self._fieldmodule.findFieldByName(self._modelCoordinatesFieldName)
        else:
            mesh = self.getMeshHighestDimension()
            element = mesh.createElementiterator().next()
            if element.isValid():
                fieldcache = self._fieldmodule.createFieldcache()
                fieldcache.setElement(element)
                fielditer = self._fieldmodule.createFielditerator()
                field = fielditer.next()
                while field.isValid():
                    if field.isTypeCoordinate() and (field.getNumberOfComponents() == 3) and \
                            (field.castFiniteElement().isValid()):
                        if field.isDefinedAtLocation(fieldcache):
                            break
                    field = fielditer.next()
                else:
                    field = None
        if field:
            self.setModelCoordinatesField(field, clear=False)

    def _discoverModelFitGroup(self):
        """
        Find model fit group by current name or use defaylt None.
        """
        self._modelFitGroup = None
        fieldGroup = None
        if self._modelFitGroupName:
            fieldGroup = self._fieldmodule.findFieldByName(self._modelFitGroupName).castGroup()
            if not fieldGroup.isValid():
                fieldGroup = None
        self._modelFitGroup = fieldGroup
        if not fieldGroup:
            self._modelFitGroupName = None

    def _defineDataEmbedding(self):
        """
        Defines self._dataHostCoordinatesField to give the value of self._modelCoordinatesField at
        embedded location self._dataHostLocationField. Also self._dataHostDeltaCoordinatesField for
        visualising difference from self._dataCoordinatesField.
        Need to call again if self._modelCoordinatesField is changed.
        """
        if not (self._modelCoordinatesField and self._dataCoordinatesField):
            return  # on first load, can't call until setModelCoordinatesField and setDataCoordinatesField
        with ChangeManager(self._fieldmodule):
            # Find mesh location at nearest point in mesh
            # workaround nearest not working on 3D elements by finding nearest on boundary if exact not found
            # Note: not finding exact/nearest on self._modelFitGroup; I assume it contains all found locations
            mesh = self.getMeshHighestDimension()
            meshDimension = mesh.getDimension()
            dataFindHostLocation = self._fieldmodule.createFieldFindMeshLocation(
                self._dataCoordinatesField, self._modelCoordinatesField, mesh)
            if meshDimension < 3:
                dataFindHostLocation.setSearchMode(FieldFindMeshLocation.SEARCH_MODE_NEAREST)
            else:
                dataFindHostLocationExact = dataFindHostLocation
                dataFindHostLocationExact.setSearchMode(FieldFindMeshLocation.SEARCH_MODE_EXACT)
                mesh2d = self.getMesh(2)
                boundaryGroup = self._fieldmodule.createFieldGroup()
                boundaryMeshGroup = boundaryGroup.createMeshGroup(mesh2d)
                boundaryMeshGroup.addElementsConditional(self._fieldmodule.createFieldIsExterior())
                dataFindHostBoundaryLocationNearest = self._fieldmodule.createFieldFindMeshLocation(
                    self._dataCoordinatesField, self._modelCoordinatesField, mesh)
                dataFindHostBoundaryLocationNearest.setSearchMode(FieldFindMeshLocation.SEARCH_MODE_NEAREST)
                assert RESULT_OK == dataFindHostBoundaryLocationNearest.setSearchMesh(boundaryMeshGroup)
                dataFindHostLocation = self._fieldmodule.createFieldIf(
                    self._fieldmodule.createFieldIsDefined(dataFindHostLocationExact),
                    dataFindHostLocationExact, dataFindHostBoundaryLocationNearest)
                del dataFindHostBoundaryLocationNearest
                del dataFindHostLocationExact
                del boundaryMeshGroup
                del boundaryGroup
            self._dataHostLocationField = findOrCreateFieldStoredMeshLocation(
                self._fieldmodule, mesh, "data_location_" + mesh.getName(), managed=False)
            orphanFieldByName(self._fieldmodule, "data_host_coordinates")
            orphanFieldByName(self._fieldmodule, "data_host_delta_coordinates")
            self._dataHostCoordinatesField = self._fieldmodule.createFieldEmbedded(
                self._modelCoordinatesField, self._dataHostLocationField)
            self._dataHostCoordinatesField.setName(
                getUniqueFieldName(self._fieldmodule, "data_host_coordinates"))
            self._dataHostDeltaCoordinatesField = self._dataHostCoordinatesField - self._dataHostCoordinatesField
            self._dataHostDeltaCoordinatesField.setName(
                getUniqueFieldName(self._fieldmodule, "data_host_delta_coordinates"))
            # define storage fpr host location on all data points
            datapoints = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
            nodetemplate = datapoints.createNodetemplate()
            nodetemplate.defineField(self._dataHostLocationField)
            nodeIter = datapoints.createNodeiterator()
            node = nodeIter.next()
            while node.isValid():
                node.merge(nodetemplate)
                node = nodeIter.next()
            del nodetemplate
            # assign found mesh location to stored mesh location
            fieldassignment = self._dataHostLocationField.createFieldassignment(dataFindHostLocation)
            fieldassignment.setNodeset(datapoints)
            result = fieldassignment.assign()
            if result == RESULT_WARNING_PART_DONE:
                print("FieldFitter warning: not all datapoints have model coordinates field defined on them")
            elif result != RESULT_OK:
                print("FieldFitter error: Cannot find host location for datapoints", file=sys.stderr)
            del dataFindHostLocation

    def _getFieldTimesequence(self, field: FieldFiniteElement):
        """
        Get time sequence used for field on datapoints, on None if none.
        :param field: Field to query at datapoints.
        :return: Timesequence or None.
        """
        datapoints = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        datapoint = datapoints.createNodeiterator().next()
        if datapoint.isValid():
            dataNodetemplate = datapoints.createNodetemplate()
            result = dataNodetemplate.defineFieldFromNode(field, datapoint)
            if result == RESULT_OK:
                timesequence = dataNodetemplate.getTimesequence(field)
                if timesequence.isValid():
                    return timesequence;
        return None

    def getFieldTimeCount(self, name: str) -> int:
        """
        Get the number of times held for field data.
        :param name:
        :return: Number of times, or 0 if not time-varying.
        """
        field = self._fieldmodule.findFieldByName(name).castFiniteElement()
        timesequence = self._getFieldTimesequence(field)
        if timesequence:
            return timesequence.getNumberOfTimes()
        return 0

    def getFieldTimes(self, name: str) -> list:
        """
        Get list of times parameters are held for field of name.
        :param name: Name of field to query.
        :return: List of times.
        """
        field = self._fieldmodule.findFieldByName(name).castFiniteElement()
        times = []
        timesequence = self._getFieldTimesequence(field)
        if timesequence:
            timeCount = timesequence.getNumberOfTimes()
            for timeIndex in range(1, timeCount + 1):
                times.append(timesequence.getTime(timeIndex))
        return times

    def _defineField(self, field: FieldFiniteElement) -> bool:
        """
        Define field over modelFitGroup or whole mesh if None.
        Use first component of model coordinates field as template for all components of field.
        :param field: Finite element field to define.
        :return: True on success, otherwise False.
        """
        mesh = self.getMeshHighestDimension()
        nodes = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        meshGroup = mesh
        nodesetGroup = nodes
        if self._modelFitGroup:
            meshGroup = self._modelFitGroup.getMeshGroup(mesh)
            if (not meshGroup.isValid()) or (meshGroup.getSize() == 0):
                print("Model fit group mesh is empty")
                return False
            nodesetGroup = self._modelFitGroup.getNodesetGroup(nodes)
            if (not nodesetGroup.isValid()) or (nodesetGroup.getSize() == 0):
                print("Model fit group nodeset is empty")
                return False
        timesequence = self._getFieldTimesequence(field)
        with ChangeManager(self._fieldmodule):
            # Define over nodes
            coordinateNodetemplate = nodesetGroup.createNodetemplate()
            valueVersions = []
            nodetemplate = nodesetGroup.createNodetemplate()
            nodetemplate.defineField(field)
            componentCount = field.getNumberOfComponents()
            zero = [0.0]*componentCount
            fieldcache = self._fieldmodule.createFieldcache()
            nodeValueLabelsCount = len(self._nodeValueLabels)
            nodeIter = nodesetGroup.createNodeiterator()
            node = nodeIter.next()
            while node.isValid():
                result = coordinateNodetemplate.defineFieldFromNode(self._modelCoordinatesField, node)
                if result == RESULT_OK:
                    coordinateValueVersions = [coordinateNodetemplate.getValueNumberOfVersions(
                        self._modelCoordinatesField, 1, self._nodeValueLabels[i]) for i in range(nodeValueLabelsCount)]
                    if valueVersions != coordinateValueVersions:
                        valueVersions = coordinateValueVersions
                        for i in range(nodeValueLabelsCount):
                            nodetemplate.setValueNumberOfVersions(field, -1, self._nodeValueLabels[i], valueVersions[i])
                        if timesequence:
                            nodetemplate.setTimesequence(field, timesequence)
                    if node.merge(nodetemplate) != RESULT_OK:
                        print("Failed to define node field")
                        return False
                    # set parameters to zero
                    fieldcache.setNode(node)
                    for i in range(nodeValueLabelsCount):
                        for v in range(valueVersions[i]):
                            field.setNodeParameters(fieldcache, -1, self._nodeValueLabels[i], v + 1, zero)
                node = nodeIter.next()
            # Define over elements
            elementtemplate = meshGroup.createElementtemplate()
            elementIter = meshGroup.createElementiterator()
            element = elementIter.next()
            while element.isValid():
                eft = element.getElementfieldtemplate(self._modelCoordinatesField, 1)
                if eft.isValid():
                    elementtemplate.defineField(field, -1, eft)
                    if element.merge(elementtemplate) != RESULT_OK:
                        print("Failed to define element field")
                        return False
                element = elementIter.next()
        return True

    def _undefineField(self, field: FieldFiniteElement) -> bool:
        """
        Undefine field over modelFitGroup or whole mesh if None.
        :param field: Finite element field to undefine.
        """
        mesh = self.getMeshHighestDimension()
        nodes = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        meshGroup = mesh
        nodesetGroup = nodes
        if self._modelFitGroup:
            meshGroup = self._modelFitGroup.getMeshGroup(mesh)
            if (not meshGroup.isValid()) or (meshGroup.getSize() == 0):
                print("Model fit group mesh is empty")
            nodesetGroup = self._modelFitGroup.getNodesetGroup(nodes)
            if (not nodesetGroup.isValid()) or (nodesetGroup.getSize() == 0):
                print("Model fit group nodeset is empty")
        with ChangeManager(self._fieldmodule):
            # Undefine over elements
            elementtemplate = meshGroup.createElementtemplate()
            elementtemplate.undefineField(field)
            elementIter = meshGroup.createElementiterator()
            element = elementIter.next()
            while element.isValid():
                element.merge(elementtemplate)
                element = elementIter.next()
            # Undefine over nodes
            nodetemplate = nodesetGroup.createNodetemplate()
            nodetemplate.undefineField(field)
            nodeIter = nodesetGroup.createNodeiterator()
            node = nodeIter.next()
            while node.isValid():
                node.merge(nodetemplate)
                node = nodeIter.next()
            self._hasFitFields[field.getName()] = False

    def _fitField(self, field: FieldFiniteElement) -> bool:
        """
        Fit field to data, time-varying if data has a time sequence.
        Field must have been defined first with self._defineField(field).
        :param field: Finite element field to fit.
        :return: True on success, otherwise False.
        """
        diagnostic_level_1 = self.getDiagnosticLevel() > 0
        diagnostic_level_2 = self.getDiagnosticLevel() > 1
        if diagnostic_level_1:
            print("fitting field:", field.getName())
        optimisation = self._fieldmodule.createOptimisation()
        optimisation.setMethod(Optimisation.METHOD_NEWTON)
        optimisation.addDependentField(field)

        with ChangeManager(self._fieldmodule):
            dataObjective = self._createDataObjectiveField(field)
            result = optimisation.addObjectiveField(dataObjective)
            assert result == RESULT_OK, "Fit Field:  Could not add data objective field"
            gradientPenaltyObjective = None
            if any(self._gradient1Penalty) or any(self._gradient2Penalty):
                gradientPenaltyObjective = self._createGradientPenaltyObjectiveField(field)
                result = optimisation.addObjectiveField(gradientPenaltyObjective)
                assert result == RESULT_OK, "Fit Field:  Could not add gradient penalty objective field"

        timesequence = self._getFieldTimesequence(field)
        if timesequence:
            timeCount = timesequence.getNumberOfTimes()
        else:
            timeCount = 1

        objectiveFormat = "{:12e}"

        time = None
        start_epoch = None
        fit_start_epoch = None
        if diagnostic_level_1:
            fit_start_epoch = timer()

        fieldcache = self._fieldmodule.createFieldcache()
        for timeIndex in range(1, timeCount + 1):
            if diagnostic_level_1:
                start_epoch = timer()
            if timesequence:
                time = timesequence.getTime(timeIndex)
                optimisation.setAttributeReal(Optimisation.ATTRIBUTE_FIELD_PARAMETERS_TIME, time)
                fieldcache.setTime(time)

            if diagnostic_level_1:
                name = field.getName()
                if timesequence:
                    name += ", time " + str(timeIndex) + "/" + str(timeCount) + " = " + str(time)
                print("Fit field: " + name)
                result, objective = dataObjective.evaluateReal(fieldcache, 1)
                print("  BEGIN Data objective", objectiveFormat.format(objective))
                if gradientPenaltyObjective:
                    result, objective = gradientPenaltyObjective.evaluateReal(
                        fieldcache, gradientPenaltyObjective.getNumberOfComponents())
                    print("  BEGIN Gradient penalty objective", objectiveFormat.format(objective))

            result = optimisation.optimise()
            if diagnostic_level_2:
                solutionReport = optimisation.getSolutionReport()
                print(solutionReport)
            assert result == RESULT_OK, "Fit Field:  Optimisation failed with result " + str(result)

            if diagnostic_level_1:
                result, objective = dataObjective.evaluateReal(fieldcache, 1)
                print("    END Data objective", objectiveFormat.format(objective))
                if gradientPenaltyObjective:
                    result, objective = gradientPenaltyObjective.evaluateReal(
                        fieldcache, gradientPenaltyObjective.getNumberOfComponents())
                    print("    END Gradient penalty objective", objectiveFormat.format(objective))
                print(f"elapsed time: {timer() - start_epoch} (s)")
                print("--------")

        if diagnostic_level_1:
            print(f"total elapsed time: {timer() - fit_start_epoch} (s)")

        self._hasFitFields[field.getName()] = True
        return True

    def _createDataObjectiveField(self, field: FieldFiniteElement):
        """
        Get FieldNodesetSum objective for data projected onto mesh, including markers with fixed locations.
        Assumes ChangeManager(fieldmodule) is in effect.
        :return: Zinc FieldNodesetSum.
        """
        hostField = self._fieldmodule.createFieldEmbedded(field, self._dataHostLocationField)
        delta = hostField - field
        deltaSq = self._fieldmodule.createFieldDotProduct(delta, delta)
        datapoints = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        dataProjectionObjective = self._fieldmodule.createFieldNodesetSum(deltaSq, datapoints)
        dataProjectionObjective.setElementMapField(self._dataHostLocationField)
        return dataProjectionObjective

    def _createGradientPenaltyObjectiveField(self, field: FieldFiniteElement):
        """
        Only call for non-zero gradient{1|2}penalty values.
        Assumes ChangeManager(fieldmodule) is in effect.
        :return: Zinc field.
        """
        numberOfGaussPoints = 3
        mesh = self.getMeshHighestDimension()
        componentCount = field.getNumberOfComponents()
        dimension = mesh.getDimension()
        coordinatesCount = self._modelCoordinatesField.getNumberOfComponents()
        gradient1 = gradient1raw =\
            self._fieldmodule.createFieldGradient(field, self._modelCoordinatesField)
        fibreAxesT = None
        if self._fibreField:
            # convert to local fibre directions, with possible dimension reduction for 2D, 1D
            fibreAxes = self._fieldmodule.createFieldFibreAxes(self._fibreField, self._modelCoordinatesField)
            if not fibreAxes.isValid():
                self.printLog()
            if dimension == 3:
                fibreAxesT = self._fieldmodule.createFieldTranspose(3, fibreAxes)
            elif dimension == 2:
                fibreAxesT = self._fieldmodule.createFieldComponent(
                    fibreAxes, [1, 4, 2, 5, 3, 6] if (coordinatesCount == 3) else [1, 4, 2, 5])
            else:  # dimension == 1
                fibreAxesT = self._fieldmodule.createFieldComponent(
                    fibreAxes, [1, 2, 3] if (coordinatesCount == 3) else [1, 2] if (coordinatesCount == 2) else [1])
        gradientTerm = None
        if any(self._gradient1Penalty):
            if self._fibreField:
                gradient1 = self._fieldmodule.createFieldMatrixMultiply(componentCount, gradient1raw, fibreAxesT)
            gradient1Penalty = self.getGradient1Penalty(dimension)
            # copy for all components of field
            for c in range(1, componentCount):
                gradient1Penalty += gradient1Penalty[:dimension]
            wtSqGradient1 = self._fieldmodule.createFieldDotProduct(
                self._fieldmodule.createFieldConstant(gradient1Penalty), gradient1*gradient1)
            gradientTerm = wtSqGradient1
        if any(self._gradient2Penalty):
            # don't do gradient of gradient1 with fibres due to slow finite difference evaluation
            gradient2 = self._fieldmodule.createFieldGradient(gradient1raw, self._modelCoordinatesField)
            if self._fibreField:
                # convert to local fibre directions
                gradient2a = self._fieldmodule.createFieldMatrixMultiply(
                    componentCount*coordinatesCount, gradient2, fibreAxesT)
                # transpose each field component submatrix of gradient2a to remultiply by fibreAxesT
                if dimension == 1:
                    gradient2aT = gradient2a
                else:
                    transposeComponents = None
                    if coordinatesCount == 3:
                        transposeComponents = [1, 3, 5, 2, 4, 6] if (dimension == 2) else [1, 4, 7, 2, 5, 8, 3, 6, 9]
                    elif coordinatesCount == 2:
                        transposeComponents = [1, 3, 2, 4]
                    matrixSize = coordinatesCount*dimension
                    for c in range(1, componentCount):
                        transposeComponents += [transposeComponents[i] + c*matrixSize for i in range(matrixSize)]
                    gradient2aT = self._fieldmodule.createFieldComponent(gradient2a, transposeComponents)
                gradient2 = self._fieldmodule.createFieldMatrixMultiply(
                    componentCount*dimension, gradient2aT, fibreAxesT)
            gradient2Penalty = self.getGradient2Penalty(dimension*dimension)
            # copy for all components of field
            for c in range(1, componentCount):
                gradient2Penalty += gradient2Penalty[:dimension*dimension]
            wtSqGradient2 = self._fieldmodule.createFieldDotProduct(
                self._fieldmodule.createFieldConstant(gradient2Penalty), gradient2*gradient2)
            assert wtSqGradient2.isValid()
            gradientTerm = (gradientTerm + wtSqGradient2) if gradientTerm else wtSqGradient2
            if not gradientTerm.isValid():
                self.printLog()
                raise AssertionError("Fieldfitter: Failed to get gradient term")
        # integrate over modelFitGroup, or whole mesh if None.
        meshGroup = mesh
        if self._modelFitGroup:
            meshGroup = self._modelFitGroup.getMeshGroup(mesh)
        gradientPenaltyObjective =\
            self._fieldmodule.createFieldMeshIntegral(gradientTerm, self._modelCoordinatesField, meshGroup)
        gradientPenaltyObjective.setNumbersOfPoints(numberOfGaussPoints)
        return gradientPenaltyObjective
