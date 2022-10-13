from base64 import b64encode
import json
from nacl import encoding, public
import requests, sys, os, http

def retrieve_input(env):
  value = os.getenv(env)

  if value is None:
    sys.exit("{} is a required input and must be set.".format(env))

  return value

def generate_authentication_headers(access_token):
  return {
    "Accept": "application/vnd.github+json",
    "Authorization": "Bearer {}".format(access_token),
  }

def retrieve_public_key_details(base_url, access_token):
  response = requests.get(
    "{}/public-key".format(base_url), 
    headers=generate_authentication_headers(access_token)
  )

  if (response.status_code != http.HTTPStatus.OK):
    sys.exit(response.text)
  
  return json.loads(response.text)

def encrypt_secret(key, coding, secret_plain):
  public_key = public.PublicKey(key.encode(coding), encoding.Base64Encoder())
  sealed_box = public.SealedBox(public_key)
  encrypted = sealed_box.encrypt(secret_plain.encode(coding))
  
  return b64encode(encrypted).decode(coding)

def save_secret(base_url, access_token, key_id, secret_name, secret):
  response = requests.put(
    "{}/{}".format(base_url, secret_name),
    headers=generate_authentication_headers(access_token),
    data=json.dumps({
      "encrypted_value": secret,
      "key_id": key_id,
    })
  )

  if (response.status_code != http.HTTPStatus.CREATED):
    sys.exit(response.text)


if __name__ == "__main__":
  print('Extracting input ...')
  owner = retrieve_input('OWNER')
  repository = retrieve_input('REPOSITORY')
  access_token = retrieve_input('ACCESS_TOKEN')
  secret_name = retrieve_input('SECRET_NAME')
  secret_value = os.environ.get('SECRET_VALUE', '') # Allow users to set empty secret

  base_url = "https://api.github.com/repos/{}/{}/actions/secrets".format(owner, repository)
  coding = "utf-8"

  print("Retrieving public key for {}/{} ...".format(owner, repository))
  key = retrieve_public_key_details(base_url, access_token)

  print("Encrypting secret value ...")
  secret = encrypt_secret(key['key'], coding, secret_value)

  print("Saving secret value in github action secret {} ...".format(secret_name))
  save_secret(base_url, access_token, key['key_id'], secret_name, secret)

  print("Secret saved successfully!")
  
  
