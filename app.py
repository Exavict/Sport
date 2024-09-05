from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64

app = Flask(__name__)
# 创建 SocketIO 实例以支持 WebSocket 通信
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

# 监听客户端通过 WebSocket 发送的 'video_frame' 事件
@socketio.on('video_frame')
def handle_frame(data):
    # 从 base64 编码的数据中提取并解码视频帧
    encoded_data = data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 处理视频帧 (例如翻转帧)
    frame = cv2.flip(frame, 1)

    # 编码处理后的帧为 base64 并发送回客户端
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_frame = base64.b64encode(buffer).decode('utf-8')
    emit('response_back', {'data': 'data:image/jpeg;base64,' + str(encoded_frame)})

if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
