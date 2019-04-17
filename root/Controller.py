from source import vrep
import math

class Controller(object):
    """ docstring para Robot Controller e API intercomunicacao com Python """
    def __init__(self, name, joints):
        self.name = name
        self.joints = joints
        self.id_number = None

#############################################################################################################
    # OK
    """ inicia a conexao da API cliente em python com o servidor vrep """
    def connect(self):
        vrep.simxFinish(-1)
        client_id = vrep.simxStart('127.0.0.1', 19997, True, True, 500, 5)

        if client_id != -1: # if we connected successfully
            print ('Successfully connected to remote API server')

        print "Client ID: ", client_id

        self.id_number = client_id

    # OK
    """ para a conexao da API cliente em python com o servidor vrep """
    def close_connection(self):
        vrep.simxGetPingTime(self.id_number)

        vrep.simxFinish(self.id_number)
        print('Connection closed...')

############################################################################################################
    # OK
    """ inicia a simulacao do ambiente """
    def start_sim(self):
        code = vrep.simxStartSimulation(self.id_number, vrep.simx_opmode_oneshot)
        if code != 0 and code != 1:
            print "Error on starting sim"
        code = vrep.simxSetBooleanParameter(self.id_number, vrep.sim_boolparam_display_enabled,0, vrep.simx_opmode_oneshot)



    # OK
    """ para a simulacao do ambiente """
    def stop_sim(self):
        code = vrep.simxStopSimulation(self.id_number, vrep.simx_opmode_oneshot)
        if code != 0 and code != 1:
            print "Error on stopping sim"

##############################################################################################################
    #OK
    """ retorna um valor relacionado a um objeto da simulacao """
    def get_joint_handler(self, joint_string):

        code, handler = vrep.simxGetObjectHandle(self.id_number, self.name + joint_string, vrep.simx_opmode_blocking)
        return handler

    #OK
    """ retorna um valor em graus ou radianos (padrao graus) de uma junta de um objeto da simulacao """
    def get_joint_position(self, joint_handler, degrees = True):

        code, position_rads = vrep.simxGetJointPosition(self.id_number, joint_handler, vrep.simx_opmode_streaming)

        if degrees:
            return float('%.3f'%(position_rads*(180.0/math.pi)))
        else:
            return position_rads

    #OK
    # de 0 a 2pi
    """ move uma junta a uma certa angulacao em tempo de simulacao """
    def set_joint_position(self, joint_handler, joint_degree):

        rad_conversion = joint_degree*(math.pi/180.0)
        code = vrep.simxSetJointTargetPosition(self.id_number, joint_handler, rad_conversion, vrep.simx_opmode_oneshot)

    #OK
    """ fecha o atuador da garra """
    def gripper_close(self):
        return_code = vrep.simxSetStringSignal(self.id_number,'MicoHand','true',vrep.simx_opmode_oneshot)

    #OK
    """ abre o atuador da garra """
    def gripper_open(self):
        return_code = vrep.simxSetStringSignal(self.id_number,'MicoHand','false',vrep.simx_opmode_oneshot)

################################################################################################################

    #OK
    """ devolve uma lista de handlers dos atuadores do braco """
    def get_handlers(self, handler_strings):
        handlers = []
        for string in handler_strings:
            handlers.append( self.get_joint_handler(str(string) ))

        return handlers

    #OK
    """ devolve uma lista de valores de posicoes dos atuadores do braco """
    def get_positions(self, handlers_list):
        angles = []
        for handler in handlers_list:
            angles.append( self.get_joint_position(handler) )

        return angles


    #OK
    """ escreve os valores de uma lista nos atuadores do braco """
    def set_positions(self, handler_list, position_degrees_list):

        for index in range(0,len(handler_list)):
            self.set_joint_position(int(handler_list[index]), position_degrees_list[index])

###################################################################################################################
