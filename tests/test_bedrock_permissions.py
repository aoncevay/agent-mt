#!/usr/bin/env python3
"""
Test script to verify Bedrock permissions for IAM role.
Run this in your SageMaker notebook to check if your role has the necessary permissions.

Usage:
    # Test with model ID (default):
    python test_bedrock_permissions.py
    
    # Test with Application Inference Profile ARN:
    python test_bedrock_permissions.py --arn <arn>
    python test_bedrock_permissions.py --arn arn:aws:bedrock:us-east-1:145023110438:application-inference-profile/5bczxc9bbzmo
    
    # Or in a notebook cell:
    exec(open('test_bedrock_permissions.py').read())
    
    # Test ARN in notebook:
    test_bedrock_arn_permissions(
        region="us-east-1",
        model_arn="arn:aws:bedrock:us-east-1:145023110438:application-inference-profile/5bczxc9bbzmo",
        model_provider="anthropic"
    )
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
            
            # Offer to generate policy request document
            print(f"\n💡 TIP: Run generate_policy_request_file() to create a document you can share with your admin.")
            print(f"   Or call: generate_policy_request_file(region='{region}', model_ids=['{model_id}'])")
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


def test_bedrock_arn_permissions(region="us-east-1", model_arn="arn:aws:bedrock:us-east-1:145023110438:application-inference-profile/5bczxc9bbzmo", model_provider="anthropic"):
    """
    Test Bedrock permissions for Application Inference Profile ARNs.
    Tests InvokeModel directly with ARN as modelId (same as standard model IDs).
    
    Args:
        region: AWS region (default: us-east-1)
        model_arn: Application Inference Profile ARN to test
        model_provider: Model provider (e.g., "anthropic") - used for request formatting
    """
    print("=" * 80)
    print("Bedrock ARN (Application Inference Profile) Permissions Test")
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
    
    # Step 2: Test bedrock:GetInferenceProfile (optional - just for info)
    print(f"\n[2] Testing bedrock:GetInferenceProfile for '{model_arn}' (optional)...")
    try:
        bedrock = boto3.client("bedrock", region_name=region)
        # Extract profile ID from ARN
        profile_id = model_arn.split("/")[-1]
        response = bedrock.get_inference_profile(inferenceProfileIdentifier=profile_id)
        print(f"✓ Can get inference profile details")
        print(f"  Profile Name: {response.get('inferenceProfile', {}).get('name', 'N/A')}")
        print(f"  Profile ARN: {response.get('inferenceProfile', {}).get('inferenceProfileArn', 'N/A')}")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'AccessDeniedException':
            print(f"⚠ Access denied: bedrock:GetInferenceProfile (not required for InvokeModel)")
            print(f"  Error: {e.response['Error']['Message']}")
            print(f"  → This is optional - InvokeModel may still work without it")
        elif error_code == 'ResourceNotFoundException':
            print(f"⚠ Inference profile not found")
            print(f"  Error: {e.response['Error']['Message']}")
        else:
            print(f"⚠ Error: {error_code} - {e.response['Error']['Message']}")
    except Exception as e:
        print(f"⚠ Unexpected error: {e}")
    
    # Step 3: Test bedrock:InvokeModel with ARN as modelId (the critical one)
    # This is the same as testing with a regular model ID - just use ARN as modelId
    print(f"\n[3] Testing bedrock:InvokeModel with ARN '{model_arn}' (using ARN as modelId)...")
    try:
        bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)
        
        # Prepare a simple test request
        test_prompt = "Hello, this is a test. Please respond with 'OK'."
        
        # Format for Claude models (anthropic provider)
        if model_provider == "anthropic":
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
        else:
            # Generic format for other providers
            body = json.dumps({
                "messages": [
                    {
                        "role": "user",
                        "content": test_prompt
                    }
                ]
            })
        
        # Use ARN as modelId (same as regular model IDs - this is how it works!)
        response = bedrock_runtime.invoke_model(
            modelId=model_arn,  # ARN works directly as modelId
            body=body,
            contentType="application/json",
            accept="application/json"
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        print(f"✓ Can invoke model with ARN successfully!")
        
        # Extract response text based on provider
        if model_provider == "anthropic":
            response_text = response_body.get('content', [{}])[0].get('text', 'N/A')
        else:
            response_text = str(response_body)[:50]
        
        print(f"  Response: {response_text[:50]}...")
        print(f"\n✅ ARN InvokeModel permission check passed!")
        print(f"   Your role has bedrock:InvokeModel permission for: {model_arn}")
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'AccessDeniedException':
            print(f"✗ Access denied: bedrock:InvokeModel")
            print(f"  Error: {e.response['Error']['Message']}")
            print(f"\n❌ PERMISSION ISSUE DETECTED")
            print(f"\nYour IAM role '{identity['Arn']}' needs the following permission:")
            print(f"\n1. bedrock:InvokeModel")
            print(f"   Resource: {model_arn}")
            print(f"\nNote: bedrock:GetInferenceProfile is optional and not required for InvokeModel")
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
                        "Resource": model_arn
                    }
                ]
            }, indent=2))
        elif error_code == 'ValidationException':
            print(f"⚠ Validation error (might be model-specific format issue)")
            print(f"  Error: {e.response['Error']['Message']}")
        elif error_code == 'ResourceNotFoundException':
            print(f"⚠ Inference profile not found or not accessible")
            print(f"  Error: {e.response['Error']['Message']}")
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
    import sys
    
    # Check if testing ARN
    if len(sys.argv) > 1 and sys.argv[1] == "--arn":
        # Test with ARN
        if len(sys.argv) > 2:
            arn = sys.argv[2]
        else:
            # Default ARN from vars.py
            arn = "arn:aws:bedrock:us-east-1:145023110438:application-inference-profile/5bczxc9bbzmo"
        
        provider = "anthropic"  # Default provider
        if len(sys.argv) > 3:
            provider = sys.argv[3]
        
        test_bedrock_arn_permissions(region="us-east-1", model_arn=arn, model_provider=provider)
    else:
        # Test with default Claude model ID
        test_bedrock_permissions(region="us-east-1", model_id="anthropic.claude-3-7-sonnet-20250219-v1:0")
        
        # Uncomment to test with a different model:
        # test_bedrock_permissions(region="us-east-1", model_id="qwen.qwen3-32b-v1:0")

