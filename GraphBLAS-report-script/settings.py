from datetime import date
import time

# TIME_PREFIX is a list of arguments for calling the command /usr/bin/time under python subprocess.
# Most likely, you only need to change the argument after '-f' for formatting output in another way.
TIME_PREFIX = ["/usr/bin/time", "-o", "time.txt", "-f", "{'User time': %U, 'System time': %S, 'Elapsed time': %e, 'Max memory': %M, 'CPU usage': %P}"]

# REPORT_DATE formatting name for output table if it wasn't passed as an argument.
# The way this string formatted should be self-explanatory.
REPORT_DATE = f'Report_{date.today().month}_{date.today().day}_{date.today().year}_{time.strftime("%H.%M.%S", time.localtime())}.xlsx'

# Backends for testing and paths to them.
# At the moment there is no way to set backends at the time of the script call, they must be hard-coded here.
backends = {
    'SuiteSparce:GraphBLAS' : 'algs/ssgb/build',
    'GB_Kun' : 'algs/gb_kun/build'
    }