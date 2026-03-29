@echo off
echo ========================================
echo    TensorBoard  Running...
echo ========================================
echo.

REM 가장 최근 결과 폴더의 로그를 보려면:
REM tensorboard --logdir="./ppo_results_v101/custom_ppo_logs"

REM 여러 실험을 비교하려면 (name1:path1,name2:path2 형식):
REM tensorboard --logdir="v100:./ppo_results_v100/custom_ppo_logs,v101:./ppo_results_v101/custom_ppo_logs"

REM 모든 PPO 결과를 한번에 보기:
tensorboard --logdir="."

echo.
echo TensorBoard가 실행되었습니다!
echo 브라우저에서 http://localhost:6006 을 열어주세요.
echo.
echo 종료하려면 Ctrl+C를 누르세요.
echo ========================================

pause