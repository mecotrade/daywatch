from onvif import ONVIFCamera

from time import sleep
from threading import Thread
import re


class PublicONVIFCamera(ONVIFCamera):

    def __init__(self, host, port, user, passwd):
        super().__init__(host, port, user, passwd)

    def get_definition(self, name, portType=None):
        xaddr, wsdlpath, binding_name = super().get_definition(name, portType)

        # replace IP:PORT, for example, 192.168.0.112:8899 with actual HOST:PORT
        # they might not match if ONVIF capable device is used over the public network
        return re.sub(r'[0-9]+(?:\.[0-9]+){3}:[0-9]+', '%s:%d' % (self.host, self.port), xaddr), wsdlpath, binding_name


class ONVIFConnector:

    def __init__(self, host, port, login, password, logger):

        self.logger = logger

        cam = PublicONVIFCamera(host, port, login, password)
        logger.info('connected to ONVIF camera')

        media = cam.create_media_service()
        logger.info('created media service object')

        # Get target profile
        media_profile = media.GetProfiles()[0]

        self.ptz = cam.create_ptz_service()
        logger.info('created PTZ service object')

        self.continuous_move_request = self.ptz.create_type('ContinuousMove')
        self.continuous_move_request.ProfileToken = media_profile.token
        if self.continuous_move_request.Velocity is None:
            self.continuous_move_request.Velocity = self.ptz.GetStatus({'ProfileToken': media_profile.token}).Position

        self.stop_request = self.ptz.create_type('Stop')
        self.stop_request.ProfileToken = media_profile.token

    def stop(self):
        self.logger.debug('stop requested')
        self.stop_request.PanTilt = True
        self.stop_request.Zoom = True
        self.ptz.Stop(self.stop_request)
        self.logger.debug('camera stopped')

    def continuous_move(self, pan_velocity, tilt_velocity):
        self.logger.debug('move requested with pan: %.1f, tilt: %.1f' % (pan_velocity, tilt_velocity))
        self.continuous_move_request.Velocity.PanTilt.x = pan_velocity
        self.continuous_move_request.Velocity.PanTilt.y = tilt_velocity
        self.ptz.ContinuousMove(self.continuous_move_request)
        self.logger.debug('continuous move started')

    def perform_move(self, pan_velocity, tilt_velocity, timeout):
        self.logger.debug('move pan: %.1f, tilt: %.1f, time: %.1f' % (pan_velocity, tilt_velocity, timeout))
        self.continuous_move(pan_velocity, tilt_velocity)
        sleep(timeout)
        self.stop()

    def perform_async_move(self, pan_velocity, tilt_velocity, timeout):
        move_thread = Thread(target=self.perform_move, args=(pan_velocity, tilt_velocity, timeout))
        move_thread.start()
