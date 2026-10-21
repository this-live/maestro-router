from .response import Response
from .router import Router

router = Router()
USERS = {1: {"id": 1, "name": "ada"}, 7: {"id": 7, "name": "grace"}}


@router.route("GET", "/health")
def health():
    return {"ok": True}


@router.route("GET", "/users")
def list_users():
    return {"users": sorted(USERS)}


@router.route("GET", "/users/<int:user_id>")
def get_user(user_id):
    if user_id not in USERS:
        return Response(404, {"error": "no such user"})
    return USERS[user_id]


@router.route("DELETE", "/users/<int:user_id>")
def delete_user(user_id):
    return Response(204)


@router.route("PUT", "/users/<int:user_id>")
def put_user(user_id):
    return {"updated": user_id}


@router.route("GET", "/users/me")
def me():
    return {"id": 1, "name": "ada", "me": True}


@router.route("GET", "/files/<name>")
def get_file(name):
    return {"file": name}


@router.route("GET", "/files/index")
def file_index():
    return {"files": ["a.txt", "b.txt"]}


@router.route("GET", "/teams/<team>/members/<int:member_id>")
def team_member(team, member_id):
    return {"team": team, "member": member_id}
