#!/usr/bin/env python3
"""
Test script to verify Bedrock permissions for IAM role.
Run this in your SageMaker notebook to check if your role has the necessary permissions.

Usage:
    python test_bedrock_permissions.py
    # Or in a notebook cell:
    exec(open('test_bedrock_permissions.py').read())
"""

import boto3
import json
from botocore.exceptions import ClientError

def test_bedrock_permissions(region="us-east-1", model_id="anthropic.claude-3-7-sonnet-20250219-v1:0"):
    """
    Test Bedrock permissions step by step.
    
    Args:
        region: AWS region (default: us-east-1)
        model_id: Model ID to test (default: Claude 3.7 Sonnet)
    """
    print("=" * 80)
    print("Bedrock Permissions Test")
    print("=" * 80)
    
    # Step 1: Check identity
    print("\n[1] Checking AWS Identity...")
    try:
        sts = boto3.client("sts")
        identity = sts.get_caller_identity()
        print(f"✓ Identity check passed")
        print(f"  Account: {identity['Account']}")
        print(f"  ARN: {identity['Arn']}")
        print(f"  User ID: {identity['UserId']}")
    except Exception as e:
        print(f"✗ Identity check failed: {e}")
        return False
    
    # Step 2: Test bedrock:ListFoundationModels
    print("\n[2] Testing bedrock:ListFoundationModels...")
    try:
        bedrock = boto3.client("bedrock", region_name=region)
        response = bedrock.list_foundation_models()
        model_count = len(response.get('modelSummaries', []))
        print(f"✓ Can list foundation models ({model_count} models found)")
        
        # Check if our target model is in the list
        model_ids = [m['modelId'] for m in response.get('modelSummaries', [])]
        if model_id in model_ids:
            print(f"✓ Target model '{model_id}' is available")
        else:
            print(f"⚠ Target model '{model_id}' not found in list")
            print(f"  Available Claude models:")
            claude_models = [m for m in model_ids if 'claude' in m.lower()]
            for m in claude_models[:5]:  # Show first 5
                print(f"    - {m}")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'AccessDeniedException':
            print(f"✗ Access denied: bedrock:ListFoundationModels")
            print(f"  Error: {e.response['Error']['Message']}")
            print(f"  → Your role needs 'bedrock:ListFoundationModels' permission")
        else:
            print(f"✗ Error: {error_code} - {e.response['Error']['Message']}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False
    
    # Step 3: Test bedrock:GetFoundationModel
    print(f"\n[3] Testing bedrock:GetFoundationModel for '{model_id}'...")
    try:
        bedrock = boto3.client("bedrock", region_name=region)
        response = bedrock.get_foundation_model(modelIdentifier=model_id)
        print(f"✓ Can get foundation model details")
        print(f"  Model Name: {response.get('modelDetails', {}).get('modelName', 'N/A')}")
        print(f"  Provider: {response.get('modelDetails', {}).get('providerName', 'N/A')}")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'AccessDeniedException':
            print(f"✗ Access denied: bedrock:GetFoundationModel")
            print(f"  Error: {e.response['Error']['Message']}")
            print(f"  → Your role needs 'bedrock:GetFoundationModel' permission")
        elif error_code == 'ResourceNotFoundException':
            print(f"⚠ Model not found (might not be enabled in your account)")
        else:
            print(f"✗ Error: {error_code} - {e.response['Error']['Message']}")
    except Exception as e:
        print(f"⚠ Unexpected error: {e}")
    
    # Step 4: Test bedrock:InvokeModel (the critical one)
    print(f"\n[4] Testing bedrock:InvokeModel for '{model_id}'...")
    try:
        bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)
        
        # Prepare a simple test request
        test_prompt = "Hello, this is a test. Please respond with 'OK'."
        
        # Format for Claude models
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": test_prompt
                }
            ]
        })
        
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=body,
            contentType="application/json",
            accept="application/json"
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        print(f"✓ Can invoke model successfully!")
        print(f"  Response: {response_body.get('content', [{}])[0].get('text', 'N/A')[:50]}...")
        print(f"\n✅ All permission checks passed! Your role has the necessary Bedrock permissions.")
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'AccessDeniedException':
            print(f"✗ Access denied: bedrock:InvokeModel")
            print(f"  Error: {e.response['Error']['Message']}")
            print(f"\n❌ PERMISSION ISSUE DETECTED")
            print(f"\nYour IAM role '{identity['Arn']}' needs the following permissions:")
            print(f"\n1. bedrock:InvokeModel")
            print(f"   Resource: arn:aws:bedrock:{region}::foundation-model/{model_id}")
            print(f"\n2. (Optional but recommended) bedrock:ListFoundationModels")
            print(f"   Resource: *")
            print(f"\n3. (Optional but recommended) bedrock:GetFoundationModel")
            print(f"   Resource: *")
            print(f"\nExample IAM policy:")
            print(json.dumps({
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "bedrock:InvokeModel",
                            "bedrock:InvokeModelWithResponseStream"
                        ],
                        "Resource": f"arn:aws:bedrock:{region}::foundation-model/{model_id}"
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "bedrock:ListFoundationModels",
                            "bedrock:GetFoundationModel"
                        ],
                        "Resource": "*"
                    }
                ]
            }, indent=2))
        elif error_code == 'ValidationException':
            print(f"⚠ Validation error (might be model-specific format issue)")
            print(f"  Error: {e.response['Error']['Message']}")
        elif error_code == 'ResourceNotFoundException':
            print(f"⚠ Model not found or not enabled")
            print(f"  Error: {e.response['Error']['Message']}")
            print(f"  → Enable the model in Bedrock console: AWS Console → Bedrock → Foundation models")
        else:
            print(f"✗ Error: {error_code} - {e.response['Error']['Message']}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return False


if __name__ == "__main__":
    # Test with default Claude model
    test_bedrock_permissions(region="us-east-1", model_id="anthropic.claude-3-7-sonnet-20250219-v1:0")
    
    # Uncomment to test with a different model:
    # test_bedrock_permissions(region="us-east-1", model_id="qwen.qwen3-32b-v1:0")

