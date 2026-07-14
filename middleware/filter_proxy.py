#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LM Studio 비속어 검열 미들웨어 (필터링 프록시)

역할
  클라이언트 컴퓨터(Main.html)와 A 컴퓨터의 LM Studio 사이에 위치해,
  주고받는 대화에서 비속어를 검열(마스킹)하는 OpenAI 호환 프록시 서버.

     [클라이언트 1~5] ──► http://A의IP:8800 (이 프로그램) ──► http://localhost:1234 (LM Studio)
                       ◄── 응답도 동일하게 검열되어 반환 ◄──

사용법
  1) 비속어 목록 파일 준비 (txt 또는 csv, 여러 개 지정 가능)
       - .txt : 한 줄에 한 단어 (쉼표로 여러 개 나열해도 됨, # 으로 시작하면 주석)
       - .csv : 모든 칸의 값을 비속어로 등록
  2) A 컴퓨터에서 실행:
       python filter_proxy.py --words badwords.txt badwords.csv
     옵션:
       --port 8800                          프록시가 열릴 포트 (기본 8800)
       --upstream http://localhost:1234     LM Studio 주소 (기본값)
       --mask ○                             치환 문자 (기본 ○, 단어 길이만큼 반복)
       --block-request                      사용자 입력에 비속어가 있으면 전달하지 않고
                                            거부 메시지를 반환 (기본은 마스킹 후 전달)
  3) 각 클라이언트의 Main.html → 개발자 패널 → 멀티PC(또는 AI 연동) 주소를
       http://A의IP:8800  으로 설정하면 끝. (1234 대신 8800)

특징
  - 사용자 발화(요청)와 AI 응답 양쪽 모두 검열
  - 단어 목록 파일이 바뀌면 재시작 없이 자동 재로드 (파일 수정 시각 감지)
  - CORS 허용 → 브라우저(로컬 파일)에서 바로 접속 가능
  - /v1/models 등 나머지 API는 그대로 통과
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ── 설정 (main에서 채워짐) ─────────────────────────────
UPSTREAM = 'http://localhost:1234'
MASK_CHAR = '○'
BLOCK_REQUEST = False
WORD_FILES = []          # [(경로, 마지막 수정시각)]
BAD_WORDS = []           # 정렬된 비속어 목록 (긴 단어 우선)
BAD_RE = None            # 컴파일된 정규식
STATS = {'requests': 0, 'masked_in': 0, 'masked_out': 0, 'blocked': 0}


# ── 비속어 목록 로드 ───────────────────────────────────
def load_words():
    """txt/csv 파일들에서 비속어 목록을 (재)로드한다."""
    global BAD_WORDS, BAD_RE
    words = set()
    for i, (path, _) in enumerate(WORD_FILES):
        try:
            mtime = os.path.getmtime(path)
            WORD_FILES[i] = (path, mtime)
            ext = os.path.splitext(path)[1].lower()
            with open(path, encoding='utf-8-sig') as f:
                if ext == '.csv':
                    for row in csv.reader(f):
                        for cell in row:
                            w = cell.strip()
                            if w and not w.startswith('#'):
                                words.add(w)
                else:  # txt 등: 줄 단위 + 쉼표 허용
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        for w in line.split(','):
                            w = w.strip()
                            if w:
                                words.add(w)
        except FileNotFoundError:
            print(f'[경고] 단어 파일 없음: {path}', flush=True)
        except Exception as e:
            print(f'[경고] {path} 읽기 실패: {e}', flush=True)
    BAD_WORDS = sorted(words, key=len, reverse=True)   # 긴 단어 먼저 (부분 중복 방지)
    if BAD_WORDS:
        BAD_RE = re.compile('|'.join(re.escape(w) for w in BAD_WORDS), re.IGNORECASE)
    else:
        BAD_RE = None
    print(f'[로드] 비속어 {len(BAD_WORDS)}개 등록', flush=True)


def reload_if_changed():
    """단어 파일이 수정됐으면 자동 재로드."""
    for path, mtime in WORD_FILES:
        try:
            if os.path.getmtime(path) != mtime:
                load_words()
                return
        except OSError:
            pass


def censor(text):
    """문자열에서 비속어를 마스킹. (치환된 개수, 결과) 반환."""
    if not BAD_RE or not isinstance(text, str) or not text:
        return 0, text
    count = 0

    def repl(m):
        nonlocal count
        count += 1
        return MASK_CHAR * len(m.group(0))

    return count, BAD_RE.sub(repl, text)


def censor_messages(messages):
    """chat 메시지 배열의 content를 검열. 총 치환 수 반환."""
    total = 0
    for m in messages or []:
        c = m.get('content')
        if isinstance(c, str):
            n, m['content'] = censor(c)
            total += n
        elif isinstance(c, list):     # 조각 배열 형식 지원
            for part in c:
                if isinstance(part, dict) and isinstance(part.get('text'), str):
                    n, part['text'] = censor(part['text'])
                    total += n
    return total


# ── HTTP 프록시 ───────────────────────────────────────
class FilterProxy(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')

    def _respond(self, status, body_bytes, content_type='application/json'):
        self.send_response(status)
        self._cors()
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body_bytes)))
        self.end_headers()
        self.wfile.write(body_bytes)

    def _forward(self, method, body=None):
        """업스트림(LM Studio)으로 요청을 전달하고 (status, body) 반환."""
        url = UPSTREAM.rstrip('/') + self.path
        req = urllib.request.Request(url, data=body, method=method)
        req.add_header('Content-Type', 'application/json')
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except Exception as e:
            msg = json.dumps({'error': {'message': f'미들웨어→LM Studio 연결 실패: {e}'}}).encode()
            return 502, msg

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header('Content-Length', '0')
        self.end_headers()

    def do_GET(self):
        reload_if_changed()
        status, body = self._forward('GET')
        self._respond(status, body)

    def do_POST(self):
        reload_if_changed()
        STATS['requests'] += 1
        length = int(self.headers.get('Content-Length') or 0)
        raw = self.rfile.read(length) if length else b''

        is_chat = '/chat/completions' in self.path
        if is_chat:
            try:
                data = json.loads(raw.decode('utf-8'))
            except Exception:
                self._respond(400, json.dumps({'error': {'message': '요청 JSON 파싱 실패'}}).encode())
                return

            # ① 사용자 입력 검열
            n_in = censor_messages(data.get('messages'))
            if n_in:
                STATS['masked_in'] += n_in
                print(f'[검열] 요청에서 {n_in}건 마스킹', flush=True)
                if BLOCK_REQUEST:
                    STATS['blocked'] += 1
                    reply = json.dumps({'reply': '(부적절한 표현이 감지되어 전달되지 않았습니다. 바르고 고운 말을 사용해 주세요.)',
                                        'affection_delta': -1, 'reason': '비속어 감지', 'emotion': '화남'},
                                       ensure_ascii=False)
                    body = json.dumps({'choices': [{'message': {'role': 'assistant', 'content': reply}}]},
                                      ensure_ascii=False).encode('utf-8')
                    self._respond(200, body)
                    return

            data['stream'] = False   # 검열을 위해 스트리밍 비활성화
            raw = json.dumps(data, ensure_ascii=False).encode('utf-8')

        status, body = self._forward('POST', raw)

        # ② AI 응답 검열
        if is_chat and status == 200:
            try:
                res = json.loads(body.decode('utf-8'))
                n_out = 0
                for ch in res.get('choices') or []:
                    msg = ch.get('message') or {}
                    c = msg.get('content')
                    if isinstance(c, str):
                        n, msg['content'] = censor(c)
                        n_out += n
                if n_out:
                    STATS['masked_out'] += n_out
                    print(f'[검열] 응답에서 {n_out}건 마스킹', flush=True)
                body = json.dumps(res, ensure_ascii=False).encode('utf-8')
            except Exception:
                pass   # JSON이 아니면 원문 그대로 통과
        self._respond(status, body)

    def log_message(self, fmt, *args):
        print(f'[{time.strftime("%H:%M:%S")}] {self.address_string()} {fmt % args}', flush=True)


def main():
    global UPSTREAM, MASK_CHAR, BLOCK_REQUEST, WORD_FILES
    ap = argparse.ArgumentParser(description='LM Studio 비속어 검열 미들웨어')
    ap.add_argument('--port', type=int, default=8800, help='프록시 포트 (기본 8800)')
    ap.add_argument('--upstream', default='http://localhost:1234', help='LM Studio 주소')
    ap.add_argument('--words', nargs='+', required=True, help='비속어 목록 파일 (txt/csv, 복수 지정 가능)')
    ap.add_argument('--mask', default='○', help='치환 문자 (기본 ○)')
    ap.add_argument('--block-request', action='store_true', help='비속어 포함 요청은 전달하지 않고 거부')
    args = ap.parse_args()

    UPSTREAM = args.upstream
    MASK_CHAR = args.mask
    BLOCK_REQUEST = args.block_request
    WORD_FILES = [(p, 0.0) for p in args.words]
    load_words()

    server = ThreadingHTTPServer(('0.0.0.0', args.port), FilterProxy)
    print(f'[시작] 검열 프록시 실행 중 → http://0.0.0.0:{args.port}  (업스트림: {UPSTREAM})', flush=True)
    print(f'[안내] 클라이언트 Main.html의 서버 주소를 http://A의IP:{args.port} 로 설정하세요', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f'\n[종료] 통계: {STATS}', flush=True)


if __name__ == '__main__':
    main()
