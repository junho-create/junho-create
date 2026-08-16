import sys
from rag_doctor_agent.agent.rules import AdminRules

def main():
    if len(sys.argv) < 2:
        print("Usage: python manager_cli.py '<자연어 규칙 문장>'")
        return
    cmd = sys.argv[1]
    rules = AdminRules()
    res = rules.apply_admin_command(cmd)
    print(res)

if __name__ == "__main__":
    main()
