import ctypes
import os
import numpy as np
import time
import Sensor
import logger


def object_to_voidp(obj):
    return ctypes.cast(ctypes.pointer(ctypes.py_object(obj)), ctypes.c_void_p)
    
def voidp_to_object(voidp):
    return ctypes.cast(voidp, ctypes.POINTER(ctypes.py_object)).contents.value

class SpecSensor:
    def __init__(self, sdkFolder):
        self.opened = False
        self.logger = logger.logger()

        self.sensor=None
        
        os.environ['PATH'] = sdkFolder + ';' + os.environ['PATH']
        self.dll = ctypes.WinDLL('./libs/SpecSensor.dll')
        
        self.sensors = {}
        self.system_handle = ctypes.c_void_p(0)
        self.dll.SI_Load()
                
        self.featurecallbacks = {}
        self.featurecallbackcontexts = {}
        
        (self.devicecount,err) = self.getint('DeviceCount')
        
        self.profiles = []
        
        for i in range(0,self.devicecount):
            (profile,err) = self.getenumstringbyindex('DeviceName',i)
            self.profiles.append(profile)

    def load(self):
        self.logger.debug('SpecSensor::load')
        return self.dll.SI_Load()
    
    def unload(self):
        self.logger.debug('SpecSensor::unload')
        self.dll.SI_Unload()
        
        # Test code for unloading SpecSensor library from Python.
        #libHandle = self.dll._handle
        #del self.dll
        #kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)    
        #kernel32.FreeLibrary.argtypes = [ctypes.wintypes.HMODULE]
        #kernel32.FreeLibrary(libHandle)

        return 0
        
    def getProfiles(self):
        return self.profiles
    
    def open(self, ssp_name, autoInit = False):
        self.logger.debug('SpecSensor::open {} , autoInit={}'.format(ssp_name, autoInit))
        sensor = self.sensors.get(ssp_name)
        
        # If sensor already opened.
        if sensor:
            return (0, sensor)
        
        if ssp_name not in self.profiles:
            print('given ssp name: ' +'"'+ str(ssp_name)+'"'+' is not found')
            return -666
        
        index = self.profiles.index(ssp_name)
        handle = ctypes.c_void_p(0)
        self.system_handle = handle
        error = self.dll.SI_Open(index, ctypes.byref(handle))

        if error == 0:
            self.sensor = Sensor.Sensor(self.dll, index, ssp_name, handle)
            
            if autoInit == True:
                error = self.sensor.command(handle,'Initialize')
                
        return (error, sensor)
        
    def command(self, feature):
        self.logger.debug('SpecSensor::command {}'.format(feature))
        return self.dll.SI_Command(self.system_handle, feature)

    def isimplemented(self, feature):
        valuebool = ctypes.c_bool()
        error = self.dll.SI_IsImplemented(self.system_handle, feature, ctypes.byref(valuebool))
        self.logger.debug('SpecSensor::isimplemented {} -> {}'.format(feature, valuebool.value))
        return (valuebool.value, error)

    def isreadonly(self,feature):
        valuebool = ctypes.c_bool()
        error = self.dll.SI_IsReadOnly(self.system_handle, feature, ctypes.byref(valuebool))
        self.logger.debug('SpecSensor::isreadonly {} -> {}'.format(feature, valuebool.value))
        return (valuebool.value, error)

    def iswritable(self,feature):
        valuebool = ctypes.c_bool()
        error = self.dll.SI_IsWritable(self.system_handle, feature, ctypes.byref(valuebool))
        self.logger.debug('SpecSensor::iswritable {} -> {}'.format(feature, valuebool.value))
        return (valuebool.value, error)

    def isreadable(self,feature):
        valuebool = ctypes.c_bool()
        error = self.dll.SI_IsReadable(self.system_handle, feature, ctypes.byref(valuebool))
        self.logger.debug('SpecSensor::isreadable {} -> {}'.format(feature, valuebool.value))
        return (valuebool.value, error)
    
    def setfloat(self, feature, value):
        self.logger.debug('SpecSensor::setfloat {} <- {}'.format(feature, value))
        valueDouble = ctypes.c_double(value)
        return self.dll.SI_SetFloat(self.system_handle, feature, valueDouble)

    def getfloat(self, feature):
        valueDouble = ctypes.c_double()
        error = self.dll.SI_GetFloat(self.system_handle, feature, ctypes.byref(valueDouble))
        self.logger.debug('SpecSensor::getfloat {} -> {}'.format(feature, valueDouble.value))
        return (valueDouble.value, error)

    def setint(self, feature, value):
        self.logger.debug('SpecSensor::setint {} <- {}'.format(feature, value))
        valueint = ctypes.c_int64(value)
        return self.dll.SI_SetInt(self.system_handle, feature, valueint)

    def getint(self, feature):
        valueint = ctypes.c_int64()
        error = self.dll.SI_GetInt(self.system_handle, feature, ctypes.byref(valueint))
        self.logger.debug('SpecSensor::getint {} -> {}'.format(feature, valueint.value))
        return (valueint.value, error)

    def getintmax(self, feature):
        valueint = ctypes.c_int64()
        error = self.dll.SI_GetIntMax(self.system_handle, feature, ctypes.byref(valueint))
        self.logger.debug('SpecSensor::getintmax {} -> {}'.format(feature, valueint.value))
        return (valueint.value, error)

    def getintmin(self, feature):
        valueint = ctypes.c_int64()
        error = self.dll.SI_GetIntMin(self.system_handle, feature, ctypes.byref(valueint))
        self.logger.debug('SpecSensor::getintmin {} -> {}'.format(feature, valueint.value))
        return (valueint.value, error)
    
    def getfloatmax(self, feature):
        valueDouble = ctypes.c_double()
        error = self.dll.SI_GetFloatMax(self.system_handle, feature, ctypes.byref(valueDouble))
        self.logger.debug('SpecSensor::getfloatmax {} -> {}'.format(feature, valueDouble.value))
        return (valueDouble.value, error)

    def getfloatmin(self, feature):
        valueDouble=ctypes.c_double()
        error=self.dll.SI_GetFloatMin(self.system_handle,feature,ctypes.byref(valueDouble))
        self.logger.debug('SpecSensor::getfloatmin {} -> {}'.format(feature, valueDouble.value))
        return (valueDouble.value,error)

    def setbool(self, feature, value):
        self.logger.debug('SpecSensor::setbool {} <- {}'.format(feature, value))
        valuebool = ctypes.c_bool(value)
        return self.dll.SI_SetBool(self.system_handle, feature, valuebool)

    def getbool(self, feature):
        valuebool = ctypes.c_bool()
        error = self.dll.SI_GetBool(self.system_handle, feature, ctypes.byref(valuebool))
        self.logger.debug('SpecSensor::getbool {} -> {}'.format(feature, valuebool.value))
        return (valuebool.value, error)

    def setstring(self, feature, str_value):
        self.logger.debug('SpecSensor::setstring {} <- {}'.format(feature, str_value))
        return self.dll.SI_SetString(self.system_handle, feature, str_value)
           
    def getstring(self, feature):
        maxSize = 255
        valueStringPtr = ctypes.create_unicode_buffer(maxSize)
        error = self.dll.SI_GetString(self.system_handle, feature, valueStringPtr, maxSize)
        self.logger.debug('SpecSensor::getstring {} -> {}'.format(feature, valueStringPtr.value))
        return (valueStringPtr.value, error)
    
    def geterrorstring(self, index):
        func = self.dll.SI_GetErrorString
        func.restype = ctypes.c_wchar_p
        errStr = func(index)
        return errStr

    def getstringmaxlength(self, feature):
        value = ctypes.c_int()
        error = self.dll.SI_GetStringMaxLength(self.system_handle, feature, ctypes.byref(value))
        self.logger.debug('SpecSensor::getstringmaxlength {} -> {}'.format(feature, value.value))
        return (value.value, error)

    def getenumstringmaxlength(self, feature, index):
        index = ctypes.c_int(index)
        value = ctypes.c_int()
        error = self.dll.SI_GetStringMaxLength(self.system_handle, feature, index, ctypes.byref(value))
        self.logger.debug('SpecSensor::getenumstringmaxlength {} -> {}'.format(feature, value.value))
        return (value.value, error)

    def setenumindexbystring(self, feature, str_val):
        self.logger.debug('SpecSensor::setenumindexbystring {} <- {}'.format(feature, str_val))
        return self.dll.SI_SetEnumIndexByString(self.system_handle, feature, str_val)

    def setenumindex(self,feature,index):
        self.logger.debug('SpecSensor::setenumindex {} <- {}'.format(feature, index))
        return self.dll.SI_SetEnumIndex(self.system_handle,feature,index)

    def setenumstring(self, feature, str_val):
        self.logger.debug('SpecSensor::setenumstring {} <- {}'.format(feature, str_val))
        return self.dll.SI_SetEnumString(self.system_handle, feature, str_val)

    def getenumindex(self, feature):
        value = ctypes.c_int()
        error = self.dll.SI_GetEnumIndex(self.system_handle, feature, ctypes.byref(value))
        self.logger.debug('SpecSensor::getenumindex {} -> {}'.format(feature, value.value))
        return (value.value, error)

    def getenumcount(self, feature):
        value = ctypes.c_int()
        error = self.dll.SI_GetEnumCount(self.system_handle, feature, ctypes.byref(value))
        self.logger.debug('SpecSensor::getenumcount {} -> {}'.format(feature, value.value))
        return (value.value, error)

    def getenumstringbyindex(self,feature, index):
        index = ctypes.c_int(index)
        maxSize = 255
        valueStringPtr = ctypes.create_unicode_buffer(maxSize)
        error = self.dll.SI_GetEnumStringByIndex(self.system_handle, feature, index, valueStringPtr,maxSize)
        self.logger.debug('SpecSensor::getenumstringbyindex {} -> {}'.format(feature, valueStringPtr.value))
        return (valueStringPtr.value, error)

    def getenum(self,feature):
        (count,error) = self.getenumcount(feature)
        enumList=[]
        
        for i in range(count):
            (str,error) = self.getenumstringbyindex(feature,i)
            enumList.append(str)
        
        (enumIndex,error) = self.getenumindex(feature)
        
        self.logger.debug('Sensor::getenum {} -> {} i:{}'.format(feature, enumList, enumIndex))

        return (enumList, enumIndex, error)                       

    def isenumindexavailable(self, feature, index):
        index = ctypes.c_int(index)
        valuebool = ctypes.c_bool()
        error = self.dll.SI_IsEnumIndexAvailable(self.system_handle, feature, index, ctypes.byref(valuebool))
        self.logger.debug('SpecSensor::isenumindexavailable {} -> {}'.format(feature, valuebool.value))
        return (valuebool.value, error)

    def registerfeaturecallback(self, feature, callbackFunct, context):
        self.logger.debug('SpecSensor::registerfeaturecallback {}'.format(feature))
        if self.checkCallbackExists(feature) == True:
            return -1
        
        featureCB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_void_p)
        self.featurecallbacks[feature] = featureCB(callbackFunct)
        userdata = object_to_voidp(context)
        self.featurecallbackcontexts[feature] = userdata
        
        return self.dll.SI_RegisterFeatureCallback(self.system_handle, feature, self.featurecallbacks[feature], self.featurecallbackcontexts[feature])
        
    def unregisterfeaturecallback(self, feature):
        self.logger.debug('SpecSensor::unregisterfeaturecallback {}'.format(feature))
        if self.checkCallbackExists(feature) == False:
            return -1

        self.dll.SI_UnregisterFeatureCallback(self.system_handle, feature, self.featurecallbacks[feature])
        
        del self.featurecallbacks[feature]
        del self.featurecallbackcontexts[feature]
        
        return 0
        
    def checkCallbackExists(self, feature):
        return (feature in self.featurecallbacks)
        