#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# carla library와 같이 있어야 동작함.

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

import argparse
from queue import Queue
from queue import Empty
from matplotlib import cm

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    from PIL import Image
except ImportError:
    raise RuntimeError('cannot import PIL, make sure "Pillow" package is installed')

SENSOR_POS = {'visualize': carla.Transform(carla.Location(x=-5.5, z=3.5), carla.Rotation(pitch=-10.0)), 
            'front': carla.Transform(carla.Location(x=2.7,z=2.3), carla.Rotation(roll=0.0, pitch=0.0, yaw=0.0)),
            'left': carla.Transform(carla.Location(x=2.7, y=-0.4, z=2.3), carla.Rotation(roll=0.0, pitch=0.0, yaw=-45.0)),
            'right': carla.Transform(carla.Location(x=2.7, y=-0.4, z=2.3), carla.Rotation(roll=0.0, pitch=0.0, yaw=45.0))}

def sensor_callback(data, queue):
    """
    This simple callback just stores the data on a thread safe Python Queue
    to be retrieved from the "main thread".
    """
    queue.put(data)

def parse_sensor(data, pos, name):
    if name =='rgb':
        data.convert(cc.Raw)
    elif name == 'ss':
        data.convert(cc.CityScapesPalette)
    elif name == 'depth':
        data.convert(cc.LogarithmicDepth)
    data.save_to_disk('_out/'+pos+'/'+name+'/%08d'% data.frame)


def set_camera(world, camera_bp, pos, vehicle, args):
    """
    ego vehicle에 장착할 카메라 위치 세팅
    """
    camera_bp.set_attribute("image_size_x", str(args.width))
    camera_bp.set_attribute("image_size_y", str(args.height))

    camera = world.spawn_actor(
                blueprint = camera_bp,
                transform = pos,
                attach_to = vehicle)
    return camera


def save_info(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.time)
    world = client.load_world('Town01')
    bp_lib = world.get_blueprint_library()

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 3.0
    world.apply_settings(settings)
    print("World Setting Finished...")

    vehicle = None
    cameras = dict()
    ss_cameras = dict()
    depth_cameras = dict()
    
    try:
        vehicle_bp = bp_lib.filter("vehicle.carlamotors.carlacola")[0]
        
        vehicle = world.spawn_actor(
            blueprint=vehicle_bp,
            transform=world.get_map().get_spawn_points()[0])
        vehicle.set_autopilot(True)
        print("Vehicle Setting Finished...")
        camera_bp = None
        
        for name, pos in SENSOR_POS.items():
            camera_bp = bp_lib.filter("sensor.camera.rgb")[0]            
            cameras[name] = set_camera(world, camera_bp, pos, vehicle, args)

            ss_bp = bp_lib.filter("sensor.camera.semantic_segmentation")[0]
            ss_cameras[name] = set_camera(world, ss_bp, pos, vehicle, args)

            depth_bp = bp_lib.filter("sensor.camera.depth")[0]
            depth_cameras[name] = set_camera(world, depth_bp, pos, vehicle, args)

        print("Sensor Setting Finished...")
        
        # 서버와 정보 교환을 위한 QUeue
        i_f_queue, i_l_queue, i_r_queue = Queue(), Queue(), Queue()
        ss_f_queue, ss_l_queue, ss_r_queue = Queue(), Queue(), Queue()
        d_f_queue, d_l_queue, d_r_queue = Queue(), Queue(), Queue()
        
        cameras['front'].listen(lambda data: sensor_callback(data, i_f_queue))
        # cameras['left'].listen(lambda data: sensor_callback(data, i_l_queue))
        # cameras['right'].listen(lambda data: sensor_callback(data, i_r_queue))
                
        ss_cameras['front'].listen(lambda data: sensor_callback(data, ss_f_queue))
        # ss_cameras['left'].listen(lambda data: sensor_callback(data, ss_l_queue))
        # ss_cameras['right'].listen(lambda data: sensor_callback(data, ss_r_queue))
                
        depth_cameras['front'].listen(lambda data: sensor_callback(data, d_f_queue))
        # depth_cameras['left'].listen(lambda data: sensor_callback(data, d_l_queue))
        # depth_cameras['right'].listen(lambda data: sensor_callback(data, d_r_queue))

        print("Go...")

        for frame in range(args.frames+20):
            world.tick()
            world_frame = world.get_snapshot().frame
            
            try:
                # Get the data once it's received.
                i_f_data = i_f_queue.get(True, 1.0)
                # i_l_data = i_l_queue.get(True, 1.0)
                # i_r_data = i_r_queue.get(True, 1.0)
                
                ss_f_data = ss_f_queue.get(True, 1.0)
                # ss_l_data = ss_l_queue.get(True, 1.0)
                # ss_r_data = ss_r_queue.get(True, 1.0)
                                
                d_f_data = d_f_queue.get(True, 1.0)
                # d_l_data = d_l_queue.get(True, 1.0)
                # d_r_data = d_r_queue.get(True, 1.0)
            
            except Empty:
                print("[Warning] Some sensor data has been missed")
                continue
            
            # 초반 20 프레임은 움직이지 않음.
            if frame <= 20:
                continue
            
            assert i_f_data.frame == ss_f_data.frame == d_f_data.frame == world_frame
            # assert i_l_data.frame == ss_l_data.frame == d_l_data.frame == world_frame
            # assert i_r_data.frame == ss_r_data.frame == d_r_data.frame == world_frame
            
            
            sys.stdout.write("\r(%d/%d) Simulation: %d Camera: %d" %(frame, args.frames+20, world_frame, i_f_data.frame) + ' ')
            sys.stdout.flush()

            # 현재 차량의 속도 + 네비게이션 정보(go left, go right, straight, follow)

            
            parse_sensor(i_f_data, 'front', 'rgb')
            # parse_sensor(i_l_data, 'left', 'rgb')
            # parse_sensor(i_r_data, 'right', 'rgb')
            
            parse_sensor(ss_f_data, 'front', 'ss')
            # parse_sensor(ss_l_data, 'left', 'ss')
            # parse_sensor(ss_r_data, 'right', 'ss')            

            parse_sensor(d_f_data, 'front','depth')
            # parse_sensor(d_l_data, 'left','depth')
            # parse_sensor(d_r_data, 'right','depth')

    finally:
        # Apply the original settings when exiting.
        world.apply_settings(original_settings)

        # Destroy the actors in the scene.
        if len(cameras) > 0:
            for name, _ in cameras.items():
                cameras[name].destroy()

        if len(ss_cameras) > 0:
            for name, _ in ss_cameras.items():
                ss_cameras[name].destroy()

        if vehicle:
            vehicle.destroy()


def main():
    """Start function"""
    argparser = argparse.ArgumentParser(
        description='CARLA Sensor sync and projection tutorial')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='680x420',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '-f', '--frames',
        metavar='N',
        default=200,
        type=int,
        help='number of frames to record (default: 500)')
    argparser.add_argument(
        '--no-noise',
        action='store_true',
        help='remove the drop off and noise from the normal (non-semantic) lidar')
    argparser.add_argument(
        '-t', '--time',
        metavar='T',
        default= 10.0,
        type=float,
        help='Waitining time for Environment open')
    
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    

    try:
        for name in ['front','left', 'right']:
            for sub_name in ['rgb', 'ss','depth']:
                if os.path.exists('./_out/%s/%s'%(name, sub_name)):
                    continue
                os.makedirs('./_out/'+name+'/'+sub_name)
        print("folder made...")
        save_info(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
