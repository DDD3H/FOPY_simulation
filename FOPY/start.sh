if [pypy --versition ]; then
	echo "pypyで実行開始します"
	pypy FOPYsim.py
else
	echo "pythonで実行します。"
	python FOPYsim.py