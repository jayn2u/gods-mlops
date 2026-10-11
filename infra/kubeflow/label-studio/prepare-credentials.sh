#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s /private/path/label-studio.env\n' "$0" >&2
  exit 2
fi

credential_file=$1
if [[ -e "$credential_file" ]]; then
  printf 'refusing to overwrite an existing credential file\n' >&2
  exit 1
fi

read -r -p 'Label Studio operator username: ' operator_username
read -r -s -p 'Label Studio operator password: ' operator_password
printf '\n' >&2
if [[ -z "$operator_username" || ${#operator_password} -lt 8 || ${#operator_password} -gt 128 ]]; then
  printf 'username is required and the password must contain 8 to 128 characters\n' >&2
  exit 1
fi

user_token=$(openssl rand -hex 20)
cleanup_token=$(openssl rand -hex 32)
if [[ ${#user_token} -ne 40 || ${#cleanup_token} -lt 32 ]]; then
  printf 'credential generator returned a token outside its required length\n' >&2
  exit 1
fi

umask 077
printf 'LABEL_STUDIO_USERNAME=%s\nLABEL_STUDIO_PASSWORD=%s\nLABEL_STUDIO_USER_TOKEN=%s\nMEDIA_CLEANUP_TOKEN=%s\n' \
  "$operator_username" "$operator_password" "$user_token" "$cleanup_token" > "$credential_file"
chmod 600 "$credential_file"
unset operator_password user_token cleanup_token
printf 'private Label Studio credential file created with mode 0600\n'
