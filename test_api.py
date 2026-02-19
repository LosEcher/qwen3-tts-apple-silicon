#!/usr/bin/env python3
"""
Qwen3-TTS API Test Script
测试 API 服务器的所有端点
"""

import requests
import sys

BASE_URL = "http://localhost:8840"

def test_endpoint(name, method, path, expected_status=200, **kwargs):
    """测试单个端点"""
    url = f"{BASE_URL}{path}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, timeout=10, **kwargs)
        else:
            print(f"❌ {name}: Unknown method {method}")
            return False

        if response.status_code == expected_status:
            print(f"✅ {name}: {response.status_code}")
            return True
        else:
            print(f"⚠️  {name}: Got {response.status_code}, expected {expected_status}")
            return False
    except Exception as e:
        print(f"❌ {name}: {e}")
        return False

def main():
    print("=" * 60)
    print("Qwen3-TTS API Test Suite")
    print("=" * 60)
    print(f"Testing server at: {BASE_URL}")
    print()

    # Check server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print("✅ Server is running")
        print(f"   Status: {response.json()}")
    except Exception as e:
        print(f"❌ Server not responding: {e}")
        print("   Please start the server first: python server.py")
        sys.exit(1)

    print()
    print("-" * 60)
    print("Testing Endpoints:")
    print("-" * 60)

    results = []

    # Basic endpoints
    results.append(test_endpoint("Root", "GET", "/"))
    results.append(test_endpoint("Health", "GET", "/health"))
    results.append(test_endpoint("Models", "GET", "/v1/models"))
    results.append(test_endpoint("Voices", "GET", "/v1/voices"))
    results.append(test_endpoint("Voices (Chinese)", "GET", "/v1/voices?language=chinese"))
    results.append(test_endpoint("Voices (English)", "GET", "/v1/voices?language=english"))
    results.append(test_endpoint("Voices (Japanese)", "GET", "/v1/voices?language=japanese"))

    # Queue endpoints
    results.append(test_endpoint("Queue Status", "GET", "/v1/queue/status"))
    results.append(test_endpoint("Queue Details", "GET", "/v1/queue/details"))
    results.append(test_endpoint("Queue Clear", "POST", "/v1/queue/clear"))

    # Cache endpoints
    results.append(test_endpoint("Cache Stats", "GET", "/v1/cache/stats"))
    results.append(test_endpoint("Cache Clear", "POST", "/v1/cache/clear"))

    # Audio files endpoints
    results.append(test_endpoint("Audio Files", "GET", "/v1/audio/files"))

    # Voice management
    results.append(test_endpoint("Saved Voices", "GET", "/v1/voices/saved"))

    # TTS endpoints (now with models loaded, should succeed)
    results.append(test_endpoint(
        "TTS Speech (POST)", "POST", "/v1/audio/speech",
        expected_status=200,  # Success with model
        json={"model": "custom-0.6b", "input": "Hello", "voice": "vivian"}
    ))
    results.append(test_endpoint(
        "TTS Simple (GET)", "GET", "/tts?text=hello&voice=vivian",
        expected_status=200  # Success with model
    ))
    results.append(test_endpoint(
        "Voice Design", "POST", "/v1/audio/design",
        expected_status=500,  # VoiceDesign model not downloaded
        json={"input": "Hello", "voice_description": "Deep male voice"}
    ))

    # Summary
    print()
    print("-" * 60)
    print("Test Summary:")
    print("-" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✅ All tests passed!")
        return 0
    else:
        print(f"⚠️  {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
